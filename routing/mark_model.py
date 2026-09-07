"""
routing/mark_model.py
---------------------
Shared spiking classifiers + training pipeline for the whole experiment grid.

  A1/A2  standard net, FULL frame        (the accuracy ceiling)
  B1/B2  standard net, HAND region only  (does the hand carry the signal?)
  C1/C2  SMALL net                       (can a reduced model keep the accuracy?)

  ...1 = SL-Animals-DVS (19 signs)   ...2 = DVS128 Gesture (11 gestures)

Both datasets use their frozen splits in data/splits/*.json. SL-Animals is
subject-independent (unseen signers); DVSGesture uses its official train/test
split with a validation slice carved from train.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from spikingjelly.activation_based import neuron, layer, functional, surrogate

import dataset as D
import augment as aug

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASETS = {
    "sl":  {"n_classes": 19, "split": "sl_animals.json", "cache": "sl"},
    "dvs": {"n_classes": 11, "split": "dvsgesture.json", "cache": "dvs"},
}


def _lif(kind="lif"):
    """LIF = fixed membrane time constant. PLIF = the constant is LEARNED, which
    published work reports converges faster and needs fewer timesteps."""
    if kind == "plif":
        return neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())
    return neuron.LIFNode(surrogate_function=surrogate.ATan())


class GhostConv(nn.Module):
    """Ghost module (GhostNet, CVPR'20): produce half the feature maps with a real
    conv, then generate the other half with a CHEAP depthwise conv, and concat.
    ~2x fewer parameters for the same channel count.

    Note: this is an EFFICIENCY technique, not an accuracy one — it belongs to the
    C (small-model) story rather than to raising A/B accuracy."""

    def __init__(self, cin, cout, k=3):
        super().__init__()
        half = cout // 2
        self.primary = layer.Conv2d(cin, half, k, padding=k // 2)
        self.cheap = layer.Conv2d(half, cout - half, 3, padding=1, groups=half)

    def forward(self, x):
        p = self.primary(x)
        return torch.cat([p, self.cheap(p)], dim=2)     # [T,B,C,H,W] -> channel dim = 2


class SEWBlock(nn.Module):
    """Spike-Element-Wise residual block (Fang et al., NeurIPS 2021).

    A plain spiking network degrades past ~5 conv blocks — which is exactly where
    our `deep5` plateaus. SEW fixes this by adding the SHORTCUT TO THE OUTPUT
    SPIKES (element-wise ADD), which gives true identity mapping for binary
    activations and lets depth keep helping.

    `ghost=True` builds the block from Ghost convolutions, halving its parameters
    so the same budget buys roughly twice the depth. Ghost and SEW are each well
    established; combining them inside a spiking residual block is, as far as the
    literature search went, unexplored — and it is the direct consequence of what
    this project measured: depth helps most, Ghost helps, width hurts.
    """

    def __init__(self, cin, cout, neuron_kind="lif", ghost=False, stride=1):
        super().__init__()
        conv = (lambda i, o: GhostConv(i, o)) if ghost else \
               (lambda i, o: layer.Conv2d(i, o, 3, padding=1))
        self.body = nn.Sequential(
            conv(cin, cout), layer.BatchNorm2d(cout), _lif(neuron_kind),
            conv(cout, cout), layer.BatchNorm2d(cout), _lif(neuron_kind),
        )
        # shortcut must match shape; identity when it already does
        self.short = None
        if cin != cout:
            self.short = nn.Sequential(
                layer.Conv2d(cin, cout, 1), layer.BatchNorm2d(cout), _lif(neuron_kind))

    def forward(self, x):
        s = x if self.short is None else self.short(x)
        return self.body(x) + s          # SEW: ADD on the output spikes


class DDEStem(nn.Module):
    """Directional Difference Encoding — explicit motion-direction channels.

    WHY THIS EXISTS (from our own results, not from a paper):
    6 of DVSGesture's 11 classes are DIRECTIONAL PAIRS — "right hand wave" vs
    "left hand wave", "arm clockwise" vs "counter-clockwise". The network must
    infer direction of motion. Yet T=24 bought nothing over T=16 (-0.5), which
    says the temporal information is already present and simply is not being
    READ as direction.

    So compute direction explicitly and hand it to the network. For each of four
    shifts d, the element-wise product  x[t] * roll(x[t-1], d)  is large exactly
    where content at t coincides with content at t-1 displaced by d — i.e. where
    something moved in direction d. Four cheap correlations, zero parameters.

    Output: 2 polarity channels + 4 motion-energy channels = 6.

    This is not temporal attention (which lost 11.7 points by reweighting whole
    timesteps). It adds a spatial motion field per timestep, leaving the temporal
    dynamics to the LIF neurons where they already work.
    """

    SHIFTS = ((0, 1), (0, -1), (1, 0), (-1, 0))    # right, left, down, up

    def forward(self, x):                          # x: [T, B, 2, H, W]
        prev = torch.cat([x[:1], x[:-1]], 0)       # x[t-1], first frame repeats
        outs = [x]
        for dy, dx in self.SHIFTS:
            s = torch.roll(prev, (dy, dx), dims=(-2, -1))
            if dy: s[..., :dy, :] = 0 if dy > 0 else s[..., dy:, :].mul_(0)
            if dx: s[..., :, :dx] = 0 if dx > 0 else s[..., :, dx:].mul_(0)
            # motion energy in direction d, summed over polarity -> 1 channel
            outs.append((x * s).sum(2, keepdim=True))
        return torch.cat(outs, 2)                  # [T, B, 6, H, W]


class MotionGate(nn.Module):
    """Channel gate driven by how much motion energy the frame contains.

    Temporal attention failed (-11.7) because it reweighted ENTIRE TIMESTEPS on a
    779-sample dataset — too coarse, too many parameters. This instead gates
    CHANNELS using a single scalar per frame (mean motion energy), so it costs
    2*C parameters and cannot collapse a timestep to zero.
    """

    def __init__(self, ch, hidden=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                                nn.Linear(hidden, ch), nn.Sigmoid())

    def forward(self, x, energy):                  # x: [T,B,C,H,W], energy: [T,B,1]
        g = self.fc(energy)                        # [T, B, C]
        return x * g[..., None, None]


class MarkSEW(nn.Module):
    """Deep spiking ResNet built from SEW blocks, optionally Ghost-based.

    depth = number of SEW blocks (each block is 2 convs, so depth=8 -> 16 conv
    layers, far beyond where the plain stack degrades).
    """

    def __init__(self, n_classes=19, in_size=128, depth=8, base=16,
                 neuron_kind="lif", ghost=False, pool_to=4, dropout=0.0):
        super().__init__()
        # Stem downsamples 128 -> 32 BEFORE the residual blocks. Essential here:
        # a spiking net keeps activations for all T timesteps, so one 128x128
        # feature map costs T*B*C*128*128 floats — enough to exhaust a 6 GB card
        # after a couple of blocks. Standard ResNets downsample in the stem too.
        layers = [layer.Conv2d(2, base, 3, padding=1), layer.BatchNorm2d(base),
                  _lif(neuron_kind), layer.MaxPool2d(2),          # 128 -> 64
                  layer.MaxPool2d(2)]                              # 64  -> 32
        cin = base
        # channels double at two points; two further pools take 32 -> 8
        pools = {max(1, depth // 3), max(2, 2 * depth // 3)}
        for i in range(1, depth + 1):
            cout = cin * 2 if i in pools and cin < base * 8 else cin
            layers.append(SEWBlock(cin, cout, neuron_kind, ghost))
            cin = cout
            if i in pools:
                layers.append(layer.MaxPool2d(2))
        layers += [layer.AdaptiveAvgPool2d(pool_to), layer.Flatten()]
        if dropout > 0:
            layers.append(layer.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)
        self.feat_dim = cin * pool_to * pool_to
        self.head = layer.Linear(self.feat_dim, n_classes)
        functional.set_step_mode(self, "m")

    def forward(self, x, return_feat=False):
        f = self.trunk(x.transpose(0, 1))          # [T, B, feat_dim]
        out = self.head(f).mean(0)
        # time-averaged features: what SAFL and feature-KD both read
        return (out, f.mean(0)) if return_feat else out


class TemporalAttention(nn.Module):
    """TA-SNN-style: instead of averaging the T timesteps equally, LEARN how much
    each timestep matters. A sign's informative moments are not uniform in time."""

    def __init__(self, n_feat):
        super().__init__()
        self.score = nn.Linear(n_feat, 1)

    def forward(self, x):                    # x: [T, B, n_feat]
        w = torch.softmax(self.score(x), dim=0)          # [T, B, 1]
        return (x * w).sum(0)                            # [B, n_feat]


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
class Mark(nn.Module):
    """Standard net (~335k params at 128x128). Used by A and B."""

    def __init__(self, n_classes=19, in_size=128):
        super().__init__()
        feat = in_size // 8                       # three 2x pools
        self.net = nn.Sequential(
            layer.Conv2d(2, 16, 3, padding=1), layer.BatchNorm2d(16), _lif(), layer.MaxPool2d(2),
            layer.Conv2d(16, 32, 3, padding=1), layer.BatchNorm2d(32), _lif(), layer.MaxPool2d(2),
            layer.Conv2d(32, 64, 3, padding=1), layer.BatchNorm2d(64), _lif(), layer.MaxPool2d(2),
            layer.Flatten(),
            layer.Linear(64 * feat * feat, n_classes),
        )
        functional.set_step_mode(self, "m")

    def forward(self, x):                 # [B, T, 2, S, S]
        return self.net(x.transpose(0, 1)).mean(0)


class MarkSmall(nn.Module):
    """Small net (~30-40k params, ~8-10x smaller than Mark).

    Mark's parameters are almost entirely in its classifier head
    (64*16*16*19 = 311k of 335k). So the saving comes from pooling the feature
    map down to 4x4 before the head, not from starving the conv trunk:
        Mark      : Flatten(64x16x16=16384) -> Linear -> 311k params
        MarkSmall : AdaptiveAvgPool(4) -> Flatten(32x4x4=512) -> Linear -> ~10k

    An earlier version pooled all the way to 1x1 (16 features) and collapsed to
    ~13% accuracy on 19 classes — too few features to separate the classes.
    4x4 keeps coarse spatial layout, which sign recognition needs.
    """

    def __init__(self, n_classes=19, in_size=128):
        super().__init__()
        self.net = nn.Sequential(
            layer.Conv2d(2, 16, 3, padding=1), layer.BatchNorm2d(16), _lif(), layer.MaxPool2d(2),
            layer.Conv2d(16, 32, 3, padding=1), layer.BatchNorm2d(32), _lif(), layer.MaxPool2d(2),
            layer.Conv2d(32, 32, 3, padding=1), layer.BatchNorm2d(32), _lif(), layer.MaxPool2d(2),
            layer.AdaptiveAvgPool2d(4),           # -> [.., 32, 4, 4] = 512 features
            layer.Flatten(),
            layer.Linear(32 * 4 * 4, n_classes),
        )
        functional.set_step_mode(self, "m")

    def forward(self, x):
        return self.net(x.transpose(0, 1)).mean(0)


class MarkZoo(nn.Module):
    """One configurable trunk so every zoo entry is like-for-like — only the named
    feature under test changes.

      neuron : 'lif' | 'plif'      ghost : Ghost convs instead of plain convs
      attn   : learned temporal attention instead of mean-over-time
      depth  : 3 (default) or 4 conv blocks
      width  : channel multiplier (1.0 = 16/32/64)
    """

    def __init__(self, n_classes=19, in_size=128, neuron_kind="lif",
                 ghost=False, attn=False, depth=3, width=1.0, pool_to=4,
                 dropout=0.0):
        super().__init__()
        chans = [int(c * width) for c in (16, 32, 64, 64, 128)][:depth]
        conv = (lambda i, o: GhostConv(i, o)) if ghost else \
               (lambda i, o: layer.Conv2d(i, o, 3, padding=1))
        blocks, cin = [], 2
        for c in chans:
            blocks += [conv(cin, c), layer.BatchNorm2d(c),
                       _lif(neuron_kind), layer.MaxPool2d(2)]
            cin = c
        blocks += [layer.AdaptiveAvgPool2d(pool_to), layer.Flatten()]
        if dropout > 0:
            blocks += [layer.Dropout(dropout)]
        self.trunk = nn.Sequential(*blocks)
        n_feat = cin * pool_to * pool_to
        self.head = layer.Linear(n_feat, n_classes)
        self.attn = TemporalAttention(n_classes) if attn else None
        functional.set_step_mode(self, "m")

    def forward(self, x):                      # [B, T, 2, S, S]
        out = self.head(self.trunk(x.transpose(0, 1)))   # [T, B, n_classes]
        return self.attn(out) if self.attn is not None else out.mean(0)


def _zoo(**kw):
    return lambda n, s: MarkZoo(n, s, **kw)


MODELS = {
    # existing baselines (unchanged, so earlier results stay comparable)
    "standard": Mark,
    "small":    MarkSmall,
    # zoo — each changes ONE thing vs 'base'
    "base":     _zoo(),                                   # MarkZoo equivalent of standard
    "plif":     _zoo(neuron_kind="plif"),                 # learnable membrane constant
    "attn":     _zoo(attn=True),                          # temporal attention
    "deep":     _zoo(depth=4),                            # extra conv block
    "ghost":    _zoo(ghost=True),                         # cheap convs (efficiency arm)
    "wide":     _zoo(width=1.5),                          # more channels
    # best-guess combination, refined after the zoo comparison
    "combo":    _zoo(neuron_kind="plif", attn=True, depth=4),
    # --- round 2: built from what actually won (deep 69.0%, ghost 66.7%) ---
    "deep_ghost":  _zoo(depth=4, ghost=True),            # both winners together
    "deep5":       _zoo(depth=5),                        # even deeper
    "deep5_ghost": _zoo(depth=5, ghost=True),
    "deep_drop":   _zoo(depth=4, dropout=0.3),           # depth + regularisation
    "deep_ghost_drop": _zoo(depth=4, ghost=True, dropout=0.3),
}


class MarkMotionSEW(nn.Module):
    """DVS-specific: DDE motion channels + motion-gated Ghost-SEW trunk.

    Built for a dataset whose classes ARE directions of movement. The stem hands
    the trunk explicit motion-direction evidence; the gate lets the network scale
    features by how much motion the frame carries (a still frame between gestures
    should not contribute the same as the peak of a wave).
    """

    def __init__(self, n_classes=11, in_size=128, depth=10, base=24,
                 neuron_kind="lif", ghost=True, pool_to=4, gate=True):
        super().__init__()
        self.dde = DDEStem()
        self.stem = nn.Sequential(
            layer.Conv2d(6, base, 3, padding=1), layer.BatchNorm2d(base),
            _lif(neuron_kind), layer.MaxPool2d(2), layer.MaxPool2d(2))
        blocks, cin = [], base
        pools = {max(1, depth // 3), max(2, 2 * depth // 3)}
        for i in range(1, depth + 1):
            cout = cin * 2 if i in pools and cin < base * 4 else cin
            blocks.append(SEWBlock(cin, cout, neuron_kind, ghost))
            cin = cout
            if i in pools:
                blocks.append(layer.MaxPool2d(2))
        self.trunk = nn.Sequential(*blocks)
        self.gate = MotionGate(base) if gate else None
        self.head = nn.Sequential(layer.AdaptiveAvgPool2d(pool_to), layer.Flatten(),
                                  layer.Linear(cin * pool_to * pool_to, n_classes))
        functional.set_step_mode(self, "m")

    def forward(self, x):
        x = self.dde(x.transpose(0, 1))                 # [T,B,6,H,W]
        energy = x[:, :, 2:].mean((2, 3, 4), keepdim=False).unsqueeze(-1)
        h = self.stem(x)
        if self.gate is not None:
            h = self.gate(h, energy)
        return self.head(self.trunk(h)).mean(0)


class RecurrentSEW(nn.Module):
    """Section C: DEPTH WITHOUT PARAMETERS — one SEW block applied repeatedly.

    Our strongest finding is that depth pays (base 61.4 -> deep5 76.6 -> 12 blocks
    91.9) and our C problem is that cutting parameters costs ~7 points. Those two
    facts point at one answer: SHARE the weights across depth. Applying a single
    block k times gives a k-times-deeper computation at 1x the parameter cost —
    the network unrolls in depth the way an RNN unrolls in time.

    3 stages x `repeats` applications each. Parameters scale with 3 blocks;
    computation scales with 3*repeats.
    """

    def __init__(self, n_classes=19, in_size=128, base=16, repeats=4,
                 neuron_kind="lif", ghost=True, pool_to=4):
        super().__init__()
        self.repeats = repeats
        self.stem = nn.Sequential(
            layer.Conv2d(2, base, 3, padding=1), layer.BatchNorm2d(base),
            _lif(neuron_kind), layer.MaxPool2d(2), layer.MaxPool2d(2))
        # one shared block per stage; channels change only between stages
        self.b1 = SEWBlock(base, base, neuron_kind, ghost)
        self.up1 = layer.Conv2d(base, base * 2, 1)
        self.b2 = SEWBlock(base * 2, base * 2, neuron_kind, ghost)
        self.up2 = layer.Conv2d(base * 2, base * 4, 1)
        self.b3 = SEWBlock(base * 4, base * 4, neuron_kind, ghost)
        self.pool = layer.MaxPool2d(2)
        self.head = nn.Sequential(layer.AdaptiveAvgPool2d(pool_to), layer.Flatten(),
                                  layer.Linear(base * 4 * pool_to * pool_to, n_classes))
        functional.set_step_mode(self, "m")

    def forward(self, x):
        h = self.stem(x.transpose(0, 1))
        for _ in range(self.repeats):
            h = self.b1(h)
        h = self.pool(self.up1(h))
        for _ in range(self.repeats):
            h = self.b2(h)
        h = self.pool(self.up2(h))
        for _ in range(self.repeats):
            h = self.b3(h)
        return self.head(h).mean(0)


class GradReverse(torch.autograd.Function):
    """Identity forward, sign-flipped gradient backward.

    The trick behind adversarial feature learning: the head downstream trains
    normally to predict the subject, but the gradient reaching the trunk is
    NEGATED, so the trunk is pushed to make that prediction impossible.
    """

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class SubjectHead(nn.Module):
    """Auxiliary subject classifier used only during training (SAFL).

    WHY: every model in this project UNDERFITS (train acc below val acc) yet loses
    2.3-4.9 points from val to test. Validation shares subjects with training; test
    does not. So the bottleneck is signer-specific features, not capacity.

    This head reads pooled trunk features through a gradient-reversal layer and is
    trained to identify the signer. Reversal means the trunk learns features from
    which the signer CANNOT be identified, while the main loss keeps sign identity.

    Discarded after training -> zero inference cost, unchanged parameter counts in
    the reported tables.
    """

    def __init__(self, feat_dim, n_subjects, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(hidden, n_subjects))

    def forward(self, feat, lam):
        return self.net(GradReverse.apply(feat, lam))


def _sew(**kw):
    return lambda n, s: MarkSEW(n, s, **kw)


# --- round 3: residual architectures, to break the ~5-block depth ceiling ---
MODELS.update({
    "sew8":        _sew(depth=8),                       # plain SEW control
    "sew12":       _sew(depth=12),
    "ghostsew8":   _sew(depth=8, ghost=True),           # the proposed combination
    "ghostsew12":  _sew(depth=12, ghost=True),
    "ghostsew18":  _sew(depth=18, ghost=True),          # depth Ghost pays for
    "ghostsew12w": _sew(depth=12, ghost=True, base=24),  # wider, now that residuals
                                                         # remove the degradation
    "ghostsew12d": _sew(depth=12, ghost=True, dropout=0.2),
    # --- DVS-specific: gestures are LARGE-AMPLITUDE MOTION, not fine hand shape.
    # DVS also has more training data (915 vs 779), so it tolerates more width.
    # Wider + slightly shallower than the SL winner; pairs with higher T.
    "dvsnet":     _sew(depth=10, ghost=True, base=24),
    # --- Section C: genuinely small models, built the way the evidence says
    # (depth + Ghost), instead of the old shallow 24k net that scored 53%.
    "tiny":       _sew(depth=6,  ghost=True, base=8),
    "smallsew":   _sew(depth=8,  ghost=True, base=12),
    # --- round 4: motion-aware DVS nets, and weight-shared depth for section C ---
    "motionsew":    lambda n, s: MarkMotionSEW(n, s, depth=10, base=24),
    "motionsew_ng": lambda n, s: MarkMotionSEW(n, s, depth=10, base=24, gate=False),
    "motionsew12":  lambda n, s: MarkMotionSEW(n, s, depth=12, base=16),
    "recsew4":      lambda n, s: RecurrentSEW(n, s, base=16, repeats=4),
    "recsew6":      lambda n, s: RecurrentSEW(n, s, base=16, repeats=6),
    "recsew_tiny":  lambda n, s: RecurrentSEW(n, s, base=8,  repeats=6),
})


class MarkSEWBal(MarkSEW):
    """dvsnet with its capacity REDISTRIBUTED, not increased.

    Measured: dvsnet's last four blocks hold 337,536 of its 495,059 parameters —
    68% of the model — and they run at 8x8, where the least spatial structure is
    left. Meanwhile the blocks at 32x32 and 16x16, where the gesture actually has
    spatial extent, hold under 10%.

    So cap channel growth at base*3 instead of base*4 and spend the freed budget on
    more blocks at the higher resolutions. Same parameter budget, allocated where
    there is still something to see. This is the ONE forward-pass change in this
    round: it adds no new mechanism, which is why it is not expected to repeat the
    0/5 record of PLIF, attention, crop64, RecurrentSEW and MotionSEW.
    """

    def __init__(self, n_classes=11, in_size=128, depth=12, base=24,
                 neuron_kind="lif", ghost=True, pool_to=4):
        nn.Module.__init__(self)
        layers = [layer.Conv2d(2, base, 3, padding=1), layer.BatchNorm2d(base),
                  _lif(neuron_kind), layer.MaxPool2d(2), layer.MaxPool2d(2)]
        cin = base
        # pool LATER than MarkSEW (1/2 and 3/4 through, not 1/3 and 2/3), so more
        # blocks run at 32x32 and 16x16; channels cap at 3x base, not 4x
        pools = {max(1, depth // 2), max(2, 3 * depth // 4)}
        for i in range(1, depth + 1):
            grow = i in pools and cin < base * 3
            # GhostConv splits channels in half and uses groups=half, so cout must
            # stay EVEN — round the 1.5x growth up to the nearest even number
            cout = (int(cin * 1.5) + 1) // 2 * 2 if grow else cin
            layers.append(SEWBlock(cin, cout, neuron_kind, ghost))
            cin = cout
            if i in pools:
                layers.append(layer.MaxPool2d(2))
        layers += [layer.AdaptiveAvgPool2d(pool_to), layer.Flatten()]
        self.trunk = nn.Sequential(*layers)
        self.feat_dim = cin * pool_to * pool_to
        self.head = layer.Linear(self.feat_dim, n_classes)
        functional.set_step_mode(self, "m")


# base chosen so the parameter count MATCHES its unbalanced counterpart — the
# comparison must isolate allocation, not size:
#   dvsnet_bal 479,291 vs dvsnet 495,059  (96.8%)
#   tiny_bal    37,531 vs tiny    38,587  (97.3%)
MODELS["dvsnet_bal"] = lambda n, s: MarkSEWBal(n, s, depth=12, base=40)
MODELS["tiny_bal"] = lambda n, s: MarkSEWBal(n, s, depth=8, base=12)


# ----------------------------------------------------------------------
# caching raw events (parsing .aedat / tonic every epoch is far too slow)
# ----------------------------------------------------------------------
def _cache_dir(ds):
    p = os.path.join(D.DATA, "cache", DATASETS[ds]["cache"])
    os.makedirs(p, exist_ok=True)
    return p


def _ensure_sl(rec):
    cache = _cache_dir("sl")
    if any(f.startswith(rec + "__") for f in os.listdir(cache)):
        return
    for i, (_, cls, xs, ys, ts, ps) in enumerate(D.sl_segments(rec)):
        np.savez(os.path.join(cache, f"{rec}__s{i}__c{cls}.npz"),
                 x=xs.astype(np.int16), y=ys.astype(np.int16),
                 t=ts.astype(np.int64), p=ps.astype(np.int8))


def _ensure_dvs(train_split):
    """Cache the tonic DVSGesture samples once, as <split>__i<idx>__c<cls>.npz."""
    import tonic
    cache = _cache_dir("dvs")
    tag = "train" if train_split else "test"
    if any(f.startswith(tag + "__") for f in os.listdir(cache)):
        return
    ds = tonic.datasets.DVSGesture(save_to=D.DATA, train=train_split)
    for i in range(len(ds)):
        ev, label = ds[i]
        np.savez(os.path.join(cache, f"{tag}__i{i}__c{int(label)}.npz"),
                 x=ev["x"].astype(np.int16), y=ev["y"].astype(np.int16),
                 t=ev["t"].astype(np.int64), p=ev["p"].astype(np.int8))


def _dvs_users(train_split):
    """idx -> subject id, from tonic's own `users` list (same ordering as our
    `<split>__i<idx>__…` cache keys). Needed for subject-adversarial training."""
    import tonic
    ds = tonic.datasets.DVSGesture(save_to=D.DATA, train=train_split)
    return list(getattr(ds, "users", []))


class RecognitionDataset(Dataset):
    """One sample = one sign/gesture -> (tensor, label).

    Two speed/quality features:
      • tensor cache — the hand mask is expensive and was recomputed on EVERY
        sample load, every epoch (45s vs 26s per epoch). Now the built tensor is
        cached once per (dataset, mode, T, keep_frac) as compressed uint8.
      • augment — train split only, never val/test.
    """

    def __init__(self, ds, keys, mode, T, keep_frac, split_tag=None, augment=False,
                 with_subject=False, aug_preset=None):
        self.mode, self.T, self.keep_frac = mode, T, keep_frac
        self.augment = augment
        # 'sl' allows flips; 'dvs' does NOT. '*_subj'/'*_light' variants: augment.py
        self.aug_preset = aug_preset or ds
        self.with_subject = with_subject
        self.rng = np.random.default_rng()
        self.tdir = os.path.join(D.DATA, "cache", "tensors",
                                 f"{ds}_{mode}_T{T}_keep{keep_frac}")
        os.makedirs(self.tdir, exist_ok=True)
        cache = _cache_dir(ds)
        self.samples = []                          # (path, label, raw_subject)
        if ds == "sl":
            for rec in keys:                      # keys = recording names
                _ensure_sl(rec)
                subj = rec.split("_")[0]          # 'user00_indoor' -> 'user00'
                for f in sorted(os.listdir(cache)):
                    if f.startswith(rec + "__"):
                        cls = int(f.split("__c")[1].split(".")[0])
                        self.samples.append(
                            (os.path.join(cache, f), D.SL_TO_IDX[cls], subj))
        else:
            is_train = split_tag == "train"
            _ensure_dvs(is_train)                 # keys = sample indices
            users = _dvs_users(is_train) if with_subject else []
            want = set(int(k) for k in keys)
            for f in sorted(os.listdir(cache)):
                if not f.startswith(split_tag + "__"):
                    continue
                idx = int(f.split("__i")[1].split("__")[0])
                if idx in want:
                    cls = int(f.split("__c")[1].split(".")[0])
                    subj = users[idx] if idx < len(users) else -1
                    self.samples.append((os.path.join(cache, f), cls, subj))
        # contiguous 0..N-1 subject ids for the auxiliary classifier
        uniq = sorted({s for _, _, s in self.samples})
        self.subject_map = {s: i for i, s in enumerate(uniq)}
        self.n_subjects = len(uniq)

    def __len__(self):
        return len(self.samples)

    def _tensor(self, path):
        """Build the mode-specific tensor once, then reuse it from cache."""
        cpath = os.path.join(self.tdir, os.path.basename(path).replace(".npz", ".npy"))
        if os.path.exists(cpath):
            return np.load(cpath).astype(np.float32) / 255.0
        z = np.load(path)
        t = D.recognition_sample(z["x"].astype(int), z["y"].astype(int),
                                 z["t"].astype(float), z["p"].astype(int),
                                 T=self.T, mode=self.mode, keep_frac=self.keep_frac)
        np.save(cpath, np.clip(t * 255.0, 0, 255).astype(np.uint8))   # ~4x smaller
        return t

    def __getitem__(self, i):
        path, label, subj = self.samples[i]
        tens = self._tensor(path)
        if self.augment:
            tens = aug.augment(tens, self.rng, preset=self.aug_preset)
        x = torch.from_numpy(np.ascontiguousarray(tens))
        if self.with_subject:
            return x, label, self.subject_map[subj]
        return x, label


# ----------------------------------------------------------------------
def _run(model, loader, opt=None, n_classes=19, mix_p=0.0, smooth=0.0,
         teacher=None, kd_w=0.5, kd_T=4.0, subj_head=None, lam=0.0,
         feat_proj=None, feat_w=0.0):
    """One pass. opt=None -> eval. Returns (loss, acc).
    mix_p: probability of applying CutMix to a training batch (train only).

    teacher: optional trained model whose softened outputs the student also fits.
    A small network trained on hard labels alone gets one bit of information per
    sample; the teacher's full distribution says WHICH classes are confusable,
    which is most of what a low-capacity model is missing. kd_w blends the two
    losses, kd_T softens the teacher's distribution.
    """
    train = opt is not None
    model.train(train)
    lossf = nn.CrossEntropyLoss(label_smoothing=smooth if train else 0.0)
    rng = np.random.default_rng()
    tot_loss = correct = n = 0
    subj_correct = subj_n = 0
    for batch in loader:
        if len(batch) == 3:
            X, y, subj = batch
            subj = subj.to(DEVICE)
        else:
            (X, y), subj = batch, None
        X, y = X.to(DEVICE), y.to(DEVICE)
        mixed = train and mix_p > 0 and rng.random() < mix_p
        with torch.set_grad_enabled(train):
            functional.reset_net(model)
            if mixed:
                Xm, soft = aug.cutmix(X, y, n_classes, rng=rng)
                out = model(Xm)
                loss = -(soft * torch.log_softmax(out, 1)).sum(1).mean()
            else:
                want_feat = train and ((subj_head is not None and subj is not None)
                                       or feat_proj is not None)
                out_feat_cache = [None, None]
                if want_feat:
                    out, feat = model(X, return_feat=True)
                    out_feat_cache = [out, feat]
                    s_out = subj_head(feat, lam) if subj_head is not None else None
                    # adversarial: head learns the subject, reversed gradient makes
                    # the trunk unlearn it. Accuracy here SHOULD fall over training.
                    if s_out is not None:
                        loss_s = nn.functional.cross_entropy(s_out, subj)
                        subj_correct += (s_out.argmax(1) == subj).sum().item()
                        subj_n += len(subj)
                    else:
                        loss_s = None
                else:
                    out = model(X)
                loss = lossf(out, y)
                if want_feat and loss_s is not None:
                    loss = loss + loss_s
                if train and teacher is not None:
                    # teacher may be a single model or a LIST (ensemble). Averaging
                    # several teachers' distributions gives a lower-variance target
                    # than any one of them, which is what a small student needs.
                    tlist = teacher if isinstance(teacher, (list, tuple)) else [teacher]
                    with torch.no_grad():
                        probs, t_feat = None, None
                        for ti, tm in enumerate(tlist):
                            functional.reset_net(tm)
                            if feat_proj is not None and ti == 0:
                                t_out, t_feat = tm(X, return_feat=True)
                            else:
                                t_out = tm(X)
                            p = torch.softmax(t_out / kd_T, 1)
                            probs = p if probs is None else probs + p
                        probs = probs / len(tlist)
                    kd = nn.functional.kl_div(
                        torch.log_softmax(out / kd_T, 1), probs,
                        reduction="batchmean") * (kd_T ** 2)
                    loss = (1 - kd_w) * loss + kd_w * kd
                    if feat_proj is not None and t_feat is not None:
                        # feature-level KD: logit KD transfers WHAT the teacher
                        # decides; this transfers HOW it represents the input.
                        _, s_feat = out_feat_cache[0], out_feat_cache[1]
                        loss = loss + feat_w * nn.functional.mse_loss(
                            feat_proj(s_feat), t_feat)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()   # vs true labels even when mixed
        n += len(y)
    if subj_n:
        _run.subj_acc = subj_correct / subj_n   # diagnostic: should DROP over training
    return tot_loss / n, correct / n


def train_and_eval(code, ds, mode, net="standard", keep_frac=0.5, T=8,
                   epochs=15, batch=16, limit=None, patience=5, seed=0,
                   augment=False, mix_p=0.0, smooth=0.0, lr=1e-3, save_ckpt=True,
                   teacher_ckpt=None, kd_w=0.5, safl_lam=0.0, aug_variant=None,
                   feat_w=0.0, cv_fold=None):
    """code: label for logs/results (A1, B2, C1...). ds: 'sl'|'dvs'.
       mode: 'full'|'hand'|'crop64'. net: any key of MODELS.
       augment: train-time augmentation. mix_p: CutMix probability.
       safl_lam: subject-adversarial strength (0 = off). Ramped 0->lam over
                 training so the trunk is not destabilised before it can classify."""
    torch.manual_seed(seed)
    cfg = DATASETS[ds]
    if cv_fold is None:
        sp = json.load(open(os.path.join(D.DATA, "splits", cfg["split"])))
    else:
        # 4-fold subject-independent CV, to match the protocol published work uses
        sp = json.load(open(os.path.join(D.DATA, "splits",
                                         "sl_animals_cv4.json")))[f"fold{cv_fold}"]
    tr, va, te = sp["train"], sp["val"], sp["test"]
    if limit:
        tr, va, te = tr[:limit], va[:max(1, limit // 3)], te[:max(1, limit // 3)]

    in_size = D.CROP if mode == "crop64" else 128
    print(f"[{code}] ds={ds} mode={mode} net={net} keep_frac={keep_frac} "
          f"T={T} input={in_size}x{in_size} seed={seed} aug={augment} "
          f"mix={mix_p} device={DEVICE}")

    def mk(keys, shuf, tag, augm=False, subj=False):
        return DataLoader(RecognitionDataset(ds, keys, mode, T, keep_frac, tag, augm,
                                             with_subject=subj,
                                             aug_preset=(f'{ds}_{aug_variant}'
                                                         if augm and aug_variant else None)),
                          batch_size=batch, shuffle=shuf, num_workers=0)

    dl_tr = mk(tr, True, "train", augment, subj=safl_lam > 0)   # augment TRAIN only
    dl_va = mk(va, False, "train")              # DVS val is carved from official train
    dl_te = mk(te, False, "test" if ds == "dvs" else "train")
    print(f"samples: train {len(dl_tr.dataset)} / val {len(dl_va.dataset)} "
          f"/ test {len(dl_te.dataset)}")

    model = MODELS[net](cfg["n_classes"], in_size).to(DEVICE)
    teacher, feat_proj = None, None
    if teacher_ckpt:
        # teacher_ckpt may be a single path or a LIST of paths (ensemble teacher)
        paths = teacher_ckpt if isinstance(teacher_ckpt, (list, tuple)) else [teacher_ckpt]
        teacher = []
        for pth in paths:
            tc = torch.load(pth, map_location="cpu", weights_only=True)
            tm = MODELS[tc["net"]](tc["n_classes"], tc["in_size"])
            tm.load_state_dict(tc["state"])
            tm = tm.to(DEVICE).eval()
            for p in tm.parameters():
                p.requires_grad_(False)
            teacher.append(tm)
            print(f"teacher[{len(teacher)-1}]: {tc['net']} "
                  f"({sum(p.numel() for p in tm.parameters()):,} params) "
                  f"{os.path.basename(pth)}")
        print(f"  kd_w={kd_w}  ensemble of {len(teacher)}")
        if feat_w > 0:
            # 1x1 projection bridges the student/teacher width mismatch; it is
            # part of the TRAINING graph only and is discarded afterwards.
            feat_proj = nn.Linear(model.feat_dim, teacher[0].feat_dim).to(DEVICE)
            print(f"  feature KD on: {model.feat_dim} -> {teacher[0].feat_dim}, "
                  f"feat_w={feat_w} (projection discarded at inference)")

    subj_head = None
    if safl_lam > 0:
        n_subj = dl_tr.dataset.n_subjects
        subj_head = SubjectHead(model.feat_dim, n_subj).to(DEVICE)
        print(f"SAFL: {n_subj} training subjects, lambda ramped 0 -> {safl_lam}, "
              f"aux head {sum(p.numel() for p in subj_head.parameters()):,} params "
              f"(TRAIN ONLY — discarded at inference)")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {net} — {n_par:,} params, {cfg['n_classes']} classes")

    params = (list(model.parameters())
              + (list(subj_head.parameters()) if subj_head else [])
              + (list(feat_proj.parameters()) if feat_proj else []))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    nc = cfg["n_classes"]
    best_va, best_state, waited, history = 0.0, None, 0, []
    for ep in range(epochs):
        t0 = time.time()
        # ramp lambda 0 -> safl_lam so the classifier settles before the
        # adversary starts pulling against it
        lam = safl_lam * min(1.0, ep / max(1, epochs * 0.3))
        _run.subj_acc = None
        tl, ta = _run(model, dl_tr, opt, nc, mix_p, smooth, teacher, kd_w,
                      subj_head=subj_head, lam=lam,
                      feat_proj=feat_proj, feat_w=feat_w)
        vl, vacc = _run(model, dl_va, None, nc)
        sched.step()
        rec = {"epoch": ep, "train_loss": tl, "train_acc": ta,
               "val_loss": vl, "val_acc": vacc}
        sa = getattr(_run, "subj_acc", None)
        if sa is not None:
            rec["subj_acc"] = sa           # should FALL if SAFL is working
        history.append(rec)
        extra = f" | subj {sa:.2f} lam {lam:.2f}" if sa is not None else ""
        print(f"  ep {ep:2d}: train loss {tl:.4f} acc {ta:.3f} | "
              f"val loss {vl:.4f} acc {vacc:.3f}{extra} | {time.time()-t0:.0f}s")
        if vacc > best_va:
            best_va, best_state, waited = vacc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                print("  early stop"); break

    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_acc = _run(model, dl_te, None, nc)
    gap = max(h["train_acc"] for h in history) - best_va
    print(f"\n[{code}] BEST val {best_va:.3f} | TEST loss {test_loss:.4f} "
          f"accuracy {test_acc:.3f}  ({n_par:,} params)")
    print(f"        overfit gap (best train - best val): {gap:.3f}")
    # THE metric SAFL targets: how much is lost going to unseen subjects
    print(f"        val->test drop: {best_va - test_acc:+.3f}")
    if subj_head is not None:
        sa0 = next((h.get("subj_acc") for h in history if h.get("subj_acc")), None)
        sa1 = history[-1].get("subj_acc")
        if sa0 and sa1:
            print(f"        subject-id accuracy {sa0:.2f} -> {sa1:.2f} "
                  f"(falling = trunk is losing signer information)")

    os.makedirs(os.path.join(D.DATA, "results"), exist_ok=True)
    tag = f"{code}_{ds}_{mode}_{net}_T{T}_seed{seed}"
    if mode in ("hand", "crop64"):
        tag += f"_keep{keep_frac}"
    if augment:
        tag += "_aug"
    if save_ckpt and best_state:            # needed for the ensemble
        ck = os.path.join(D.DATA, "ckpt")
        os.makedirs(ck, exist_ok=True)
        torch.save({"state": best_state, "net": net, "ds": ds, "mode": mode,
                    "T": T, "in_size": in_size, "n_classes": nc},
                   os.path.join(ck, f"{tag}.pt"))
    with open(os.path.join(D.DATA, "results", f"{tag}.json"), "w") as f:
        json.dump({"code": code, "dataset": ds, "mode": mode, "net": net,
                   "params": n_par, "keep_frac": keep_frac, "T": T, "seed": seed,
                   "input_size": in_size, "augment": augment, "mix_p": mix_p,
                   "val_test_drop": best_va - test_acc, "safl_lam": safl_lam,
                   "aug_variant": aug_variant, "feat_w": feat_w, "cv_fold": cv_fold,
                   "n_teachers": (len(teacher) if teacher else 0),
                   "best_val_acc": best_va, "test_acc": test_acc,
                   "test_loss": test_loss, "overfit_gap": gap,
                   "history": history}, f, indent=2)
    print(f"        saved -> data/results/{tag}.json")
    return test_acc
