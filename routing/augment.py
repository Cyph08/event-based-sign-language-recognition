"""
routing/augment.py
------------------
Train-time augmentation for event tensors [T, 2, S, S].

Why this matters most: both A and B overfit badly (99% train vs ~55% val on only
779 SL-Animals samples). Published event augmentation (NDA, ECCV'22; EventMix)
reports +10-14% on similar benchmarks — a bigger gain than any architecture change.

Applied ONLY to training samples. Val/test are never augmented, otherwise the
numbers are not comparable.

    python augment.py --demo     # writes a before/after GIF so you can SEE that
                                 # the hands survive the augmentation
"""

import argparse
import os

import numpy as np
from scipy.ndimage import rotate as nd_rotate


# ----------------------------------------------------------------------
# individual transforms — all operate on [T, 2, S, S] float32
# ----------------------------------------------------------------------
def rand_flip(t, rng):
    """Mirror left-right. Signers use either hand, so a flipped sign is still valid."""
    return t[:, :, :, ::-1].copy()


def rand_shift(t, rng, max_px=8):
    """Translate in space. Makes the model robust to where the signer sits."""
    dy, dx = rng.integers(-max_px, max_px + 1, size=2)
    out = np.roll(t, (int(dy), int(dx)), axis=(2, 3))
    if dy > 0:   out[:, :, :dy, :] = 0
    elif dy < 0: out[:, :, dy:, :] = 0
    if dx > 0:   out[:, :, :, :dx] = 0
    elif dx < 0: out[:, :, :, dx:] = 0
    return out


def rand_rotate(t, rng, max_deg=12):
    """Small rotation — camera/pose tilt."""
    ang = float(rng.uniform(-max_deg, max_deg))
    return nd_rotate(t, ang, axes=(2, 3), reshape=False, order=0, mode="constant")


def rand_temporal(t, rng, max_shift=2):
    """Shift the sign in time: the gesture may start slightly earlier/later."""
    s = int(rng.integers(-max_shift, max_shift + 1))
    if s == 0:
        return t
    out = np.zeros_like(t)
    if s > 0: out[s:] = t[:-s]
    else:     out[:s] = t[-s:]
    return out


def rand_erase(t, rng, max_frac=0.25):
    """EventDrop-style: blank a random box, forcing the model not to rely on one spot."""
    S = t.shape[-1]
    h = int(rng.uniform(0.1, max_frac) * S)
    w = int(rng.uniform(0.1, max_frac) * S)
    y = int(rng.integers(0, max(1, S - h)))
    x = int(rng.integers(0, max(1, S - w)))
    out = t.copy()
    out[:, :, y:y + h, x:x + w] = 0
    return out


# ----------------------------------------------------------------------
def rand_speed(t, rng, lo=0.8, hi=1.25):
    """Resample the TIME axis — the same gesture performed faster or slower.

    Targets the measured failure mode: every model here underfits training data yet
    loses 2.3-4.9 points from validation to TEST, and test uses unseen subjects.
    Signers differ in tempo, and nothing in the existing stack simulates that
    (rand_temporal only SHIFTS the window, it does not change duration).
    """
    T = t.shape[0]
    f = rng.uniform(lo, hi)
    idx = np.clip(np.round(np.arange(T) * f), 0, T - 1).astype(int)
    return t[idx]


def rand_scale(t, rng, lo=0.85, hi=1.15):
    """Zoom about the frame centre — different body size / camera distance.

    The other axis of inter-subject variation the stack was missing. Implemented by
    cropping (zoom in) or padding (zoom out) then resampling with nearest-neighbour
    indexing, which keeps the event tensor sparse and avoids interpolation blur.
    """
    T, C, H, W = t.shape
    s = rng.uniform(lo, hi)
    nh, nw = max(8, int(H / s)), max(8, int(W / s))
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    if s >= 1.0:                                   # zoom IN: crop then stretch back
        crop = t[:, :, y0:y0 + nh, x0:x0 + nw]
    else:                                          # zoom OUT: pad then shrink back
        crop = np.zeros((T, C, nh, nw), t.dtype)
        py, px = (nh - H) // 2, (nw - W) // 2
        crop[:, :, py:py + H, px:px + W] = t
    yi = np.clip((np.arange(H) * crop.shape[2] / H).astype(int), 0, crop.shape[2] - 1)
    xi = np.clip((np.arange(W) * crop.shape[3] / W).astype(int), 0, crop.shape[3] - 1)
    return crop[:, :, yi][:, :, :, xi]


# ----------------------------------------------------------------------
# DATASET-SPECIFIC PRESETS — this matters, and getting it wrong corrupts labels
# ----------------------------------------------------------------------
# SL-Animals: a sign keeps its meaning if performed with the other hand, so a
#   left-right flip produces a valid sample of the SAME class. Flip is safe.
#
# DVSGesture: 6 of the 11 classes are DIRECTIONAL —
#   1 "Right hand wave"  vs  2 "Left hand wave"
#   3 "Right arm cw"     vs  5 "Left arm cw"
#   4 "Right arm ccw"    vs  6 "Left arm ccw"
#   Flipping a right-hand wave PRODUCES a left-hand wave while keeping the old
#   label — i.e. it teaches the model the wrong answer. Rotation likewise blurs
#   clockwise vs counter-clockwise. Both are DISABLED for this dataset.
AUG_PRESETS = {
    "sl":  dict(p_flip=0.5, p_shift=0.7, p_rot=0.3, p_time=0.5, p_erase=0.25),
    "dvs": dict(p_flip=0.0, p_shift=0.7, p_rot=0.0, p_time=0.5, p_erase=0.25),
    # "subject" variants: add tempo + body-size jitter, and REDUCE erase so the
    # stack does not simply get heavier. Train accuracy already sits BELOW val
    # accuracy in every model, so total augmentation strength is at saturation —
    # the aim is to retarget it at inter-subject variation, not add more of it.
    "sl_subj":  dict(p_flip=0.5, p_shift=0.7, p_rot=0.3, p_time=0.5, p_erase=0.15,
                     p_speed=0.5, p_scale=0.5),
    "dvs_subj": dict(p_flip=0.0, p_shift=0.7, p_rot=0.0, p_time=0.5, p_erase=0.15,
                     p_speed=0.5, p_scale=0.5),
    # lighter control: is the current stack over-regularising?
    "sl_light":  dict(p_flip=0.5, p_shift=0.7, p_rot=0.3, p_time=0.5, p_erase=0.15),
    "dvs_light": dict(p_flip=0.0, p_shift=0.7, p_rot=0.0, p_time=0.5, p_erase=0.15),
}


def augment(t, rng=None, p_flip=0.5, p_shift=0.7, p_rot=0.3, p_time=0.5,
            p_erase=0.25, p_speed=0.0, p_scale=0.0, preset=None):
    """Apply the augmentation stack to one sample.
    `preset`: key of AUG_PRESETS — 'sl'/'dvs', '*_subj' (adds tempo + size
    jitter), or '*_light' (reduced erase, the over-regularisation control)."""
    if preset is not None:
        cfg = AUG_PRESETS[preset]
        p_flip, p_shift = cfg["p_flip"], cfg["p_shift"]
        p_rot, p_time, p_erase = cfg["p_rot"], cfg["p_time"], cfg["p_erase"]
        p_speed, p_scale = cfg.get("p_speed", 0.0), cfg.get("p_scale", 0.0)
    rng = rng or np.random.default_rng()
    if p_flip and rng.random() < p_flip:  t = rand_flip(t, rng)
    if p_scale and rng.random() < p_scale: t = rand_scale(t, rng)
    if p_shift and rng.random() < p_shift: t = rand_shift(t, rng)
    if p_rot and rng.random() < p_rot:   t = rand_rotate(t, rng)
    if p_speed and rng.random() < p_speed: t = rand_speed(t, rng)
    if p_time and rng.random() < p_time:  t = rand_temporal(t, rng)
    if p_erase and rng.random() < p_erase: t = rand_erase(t, rng)
    return np.ascontiguousarray(t)


def cutmix(x, y, n_classes, alpha=1.0, rng=None):
    """EventMix-style batch mixing: paste a random box from another sample and mix
    the labels in proportion to the pasted area. Returns (x, soft_targets).
    Reported ~+12% for SNNs on DVS-Gesture."""
    import torch
    rng = rng or np.random.default_rng()
    B, T, C, H, W = x.shape
    perm = torch.randperm(B, device=x.device)
    lam = float(rng.beta(alpha, alpha))
    rh, rw = int(H * np.sqrt(1 - lam)), int(W * np.sqrt(1 - lam))
    cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
    y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    x = x.clone()
    x[:, :, :, y0:y1, x0:x1] = x[perm][:, :, :, y0:y1, x0:x1]
    lam_adj = 1 - ((y1 - y0) * (x1 - x0) / (H * W))       # true pasted fraction
    hard = torch.nn.functional.one_hot(y, n_classes).float()
    return x, lam_adj * hard + (1 - lam_adj) * hard[perm]


# ----------------------------------------------------------------------
def demo():
    """Render original vs augmented so you can check the hands survive."""
    import sys
    import imageio.v2 as imageio
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import dataset as D

    rec, cls, xs, ys, ts, ps = next(D.sl_segments())
    base = D.recognition_sample(xs, ys, ts, ps, T=8, mode="full")
    rng = np.random.default_rng(0)
    variants = [("original", base)] + [(f"aug {i+1}", augment(base.copy(), rng))
                                       for i in range(3)]
    frames = []
    for t in range(base.shape[0]):
        row = []
        for _, v in variants:
            d = v[t].sum(0)
            img = np.zeros((128, 128, 3), np.uint8)
            yy, xx = np.where(d > 0)
            img[yy, xx] = (0, 230, 0)
            row.append(img)
        frames.append(np.hstack(row))
    out = os.path.join(D.OUT, "qc", "augment_demo.gif")
    imageio.mimsave(out, frames, duration=0.25)
    print(f"demo -> {out}   panels: {', '.join(n for n, _ in variants)}")
    for name, v in variants:
        print(f"  {name:9s}: mass {v.sum():8.0f}  nonzero {100*(v>0).mean():.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        demo()
    else:
        print("pass --demo to render a before/after GIF")
