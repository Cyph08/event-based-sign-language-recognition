"""
routing/ensemble.py
-------------------
Average the predictions of several trained checkpoints.

Ensembling reliably adds a couple of points, but be careful how you report it:
"one small efficient spiking model" is a cleaner thesis than "an ensemble of
models". Report the ensemble as a separate line, not as the headline result.

    python ensemble.py --ds sl --mode full           # ensemble everything matching
    python ensemble.py --ds sl --mode full --list    # just show available checkpoints
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

import dataset as D
from mark_model import MODELS, RecognitionDataset, DATASETS, DEVICE

CKPT = os.path.join(D.DATA, "ckpt")


def load_model(path):
    # weights_only=True: our checkpoints hold only tensors + simple metadata,
    # so there is no reason to allow arbitrary unpickling here.
    c = torch.load(path, map_location="cpu", weights_only=True)
    m = MODELS[c["net"]](c["n_classes"], c["in_size"])
    m.load_state_dict(c["state"])
    return m.to(DEVICE).eval(), c


# ----------------------------------------------------------------------
# Test-time augmentation
# ----------------------------------------------------------------------
def tta_views(X, ds, n):
    """Yield n label-preserving variants of a test batch [B,T,2,S,S].

    Only transforms that CANNOT change the class are used:
      • spatial shift   — the gesture is the same wherever it sits in frame
      • temporal shift  — the gesture is the same if it starts slightly later
    Horizontal flip is used for SL-Animals only: a sign performed with the other
    hand is the same sign. For DVSGesture a flip would turn "right hand wave"
    into "left hand wave" — a DIFFERENT class — so it is never applied there.
    """
    yield X                                        # the original view
    g = torch.Generator(device="cpu").manual_seed(0)
    for i in range(n):
        v = X
        dy, dx = torch.randint(-6, 7, (2,), generator=g).tolist()
        v = torch.roll(v, (dy, dx), dims=(-2, -1))
        # zero the wrapped edges — training's rand_shift zero-fills, and TTA must
        # match it or the model sees a distortion it never trained on
        if dy > 0:   v[..., :dy, :] = 0
        elif dy < 0: v[..., dy:, :] = 0
        if dx > 0:   v[..., :, :dx] = 0
        elif dx < 0: v[..., :, dx:] = 0
        if ds == "sl" and i % 2 == 1:              # flip only where it is valid
            v = torch.flip(v, dims=[-1])
        yield v


@torch.no_grad()
def predict(model, loader, ds, n_tta=0):
    """Softmax probabilities over the test set, optionally averaged over TTA views."""
    probs, labels = [], []
    for X, y in loader:
        X = X.to(DEVICE)
        acc = None
        for v in tta_views(X, ds, n_tta):
            functional.reset_net(model)
            p = torch.softmax(model(v), 1)
            acc = p if acc is None else acc + p
        probs.append((acc / (n_tta + 1)).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def main(ds, mode, T, keep_frac, batch, pattern, n_tta, top_k):
    paths = sorted(glob.glob(os.path.join(CKPT, pattern)))
    paths = [p for p in paths if f"_{ds}_{mode}_" in os.path.basename(p)]
    # only combine checkpoints trained at the same T / input size
    paths = [p for p in paths if f"_T{T}_" in os.path.basename(p)]
    if not paths:
        print(f"no checkpoints matching ds={ds} mode={mode} T={T} in {CKPT}"); return

    # Select members by VALIDATION accuracy, never by test — choosing ensemble
    # members on test performance would make the reported number invalid.
    def val_of(p):
        r = os.path.join(D.DATA, "results",
                         os.path.basename(p).replace(".pt", ".json"))
        return json.load(open(r))["best_val_acc"] if os.path.exists(r) else -1.0

    scored = sorted(((val_of(p), p) for p in paths), reverse=True)
    if top_k:
        scored = scored[:top_k]
    paths = [p for _, p in scored]
    print(f"{len(paths)} checkpoints (TTA views: {n_tta}), "
          f"selected by VALIDATION accuracy:")
    for v, p in scored:
        print(f"    val {v:.3f}  {os.path.basename(p)}")

    sp = json.load(open(os.path.join(D.DATA, "splits", DATASETS[ds]["split"])))
    dl = DataLoader(RecognitionDataset(ds, sp["test"], mode, T, keep_frac,
                                       "test" if ds == "dvs" else "train"),
                    batch_size=batch, shuffle=False)

    total, labels, singles = None, None, []
    for p in paths:
        model, _ = load_model(p)
        plain, labels = predict(model, dl, ds, 0)
        a_plain = (plain.argmax(1) == labels).float().mean().item()
        if n_tta:
            withtta, _ = predict(model, dl, ds, n_tta)
            a_tta = (withtta.argmax(1) == labels).float().mean().item()
        else:
            withtta, a_tta = plain, a_plain
        singles.append((os.path.basename(p), a_plain, a_tta))
        total = withtta if total is None else total + withtta
        print(f"  {os.path.basename(p):52s} plain {a_plain:.3f}  +TTA {a_tta:.3f}")

    ens = (total.argmax(1) == labels).float().mean().item()
    best_plain = max(s[1] for s in singles)
    best_tta = max(s[2] for s in singles)
    print(f"\nbest single (plain)  : {best_plain:.3f}")
    print(f"best single (+TTA)   : {best_tta:.3f}   ({best_tta-best_plain:+.3f})")
    print(f"ENSEMBLE (+TTA)      : {ens:.3f}   ({ens-best_plain:+.3f} vs best plain single)")

    out = os.path.join(D.DATA, "results", f"ensemble_{ds}_{mode}_T{T}.json")
    with open(out, "w") as f:
        json.dump({"ds": ds, "mode": mode, "T": T, "n_tta": n_tta,
                   "members": [{"ckpt": s[0], "plain": s[1], "tta": s[2]}
                               for s in singles],
                   "best_single_plain": best_plain, "best_single_tta": best_tta,
                   "ensemble_acc": ens}, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="sl", choices=["sl", "dvs"])
    ap.add_argument("--mode", default="full", choices=["full", "hand", "crop64"])
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--keep_frac", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pattern", default="*.pt")
    ap.add_argument("--tta", type=int, default=4, help="extra TTA views (0 = off)")
    ap.add_argument("--top", type=int, default=4, help="keep the K best members by VALIDATION")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for p in sorted(glob.glob(os.path.join(CKPT, "*.pt"))):
            print(os.path.basename(p))
    else:
        main(a.ds, a.mode, a.T, a.keep_frac, a.batch, a.pattern, a.tta, a.top)
