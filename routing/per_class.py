"""
routing/per_class.py — per-class accuracy and confusion structure for the headline
SL-Animals model, from SAVED CHECKPOINTS (no retraining, no GPU).

WHY. Draft 1 reports one aggregate number (90.2%) and a supervisor review asked why
accuracy is high. An aggregate cannot answer that. Per-class accuracy shows whether the
model is uniformly strong or carried by a few easy signs, and the confusion matrix shows
whether the residual errors are structured (visually similar signs) or scattered (noise).

WHICH MODEL. The four SEW_ baseline seeds — 89.5 / 89.5 / 90.6 / 91.2, i.e. exactly the
90.2 +- 0.9 reported. Predictions are aggregated ACROSS ALL FOUR SEEDS rather than taken
from the best run: picking the best seed for a per-class breakdown would be selection bias
on top of an already-selected number.

CPU ONLY, deliberately. GPU training runs concurrently; a second CUDA context risks OOM on
a 6 GB card and would put an 8-hour job at risk for a two-minute analysis.

    python per_class.py
"""
import json, os, sys
import torch
from torch.utils.data import DataLoader

import dataset as D
from mark_model import MODELS, RecognitionDataset, DATASETS
from spikingjelly.activation_based import functional

torch.set_num_threads(4)                 # leave cores for the concurrent training job
CK = os.path.join(D.DATA, "ckpt")
OUT = os.path.join(D.DATA, "results", "per_class_sl.json")
SEEDS = [0, 1, 2, 3]
TAG = "SEW_sl_full_ghostsew12_T16_seed{}_aug"


def main():
    nc = DATASETS["sl"]["n_classes"]
    te = json.load(open(os.path.join(D.DATA, "splits", "sl_animals.json")))["test"]
    dl = DataLoader(RecognitionDataset("sl", te, "full", 16, 0.5, "train", False),
                    batch_size=8, shuffle=False, num_workers=0)
    print(f"test samples: {len(dl.dataset)}   classes: {nc}", flush=True)

    conf = torch.zeros(nc, nc, dtype=torch.long)      # summed over seeds: true x pred
    per_seed_cls = []                                 # [seed][class] accuracy

    for s in SEEDS:
        p = os.path.join(CK, TAG.format(s) + ".pt")
        if not os.path.exists(p):
            print(f"MISSING {p}", flush=True); continue
        ck = torch.load(p, map_location="cpu")
        model = MODELS[ck["net"]](ck["n_classes"], ck["in_size"])
        model.load_state_dict(ck["state"]); model.eval()

        hit = torch.zeros(nc); tot = torch.zeros(nc)
        with torch.no_grad():
            for xb, yb in dl:
                functional.reset_net(model)
                pred = model(xb).argmax(1)
                for t, q in zip(yb.tolist(), pred.tolist()):
                    conf[t, q] += 1
                    tot[t] += 1
                    hit[t] += (t == q)
        acc = (hit / tot.clamp(min=1))
        per_seed_cls.append(acc.tolist())
        print(f"seed {s}: overall {100*hit.sum()/tot.sum():.1f}%", flush=True)

    if not per_seed_cls:
        print("no checkpoints found", flush=True); return

    a = torch.tensor(per_seed_cls)                    # [n_seeds, n_classes]
    mean, std = a.mean(0), a.std(0) if a.shape[0] > 1 else torch.zeros(nc)

    print(f"\n=== per-class accuracy, mean over {a.shape[0]} seeds ===", flush=True)
    print(f"{'class':7s}{'n':>5s}{'mean':>9s}{'std':>8s}", flush=True)
    order = torch.argsort(mean)
    for c in order.tolist():
        print(f"{c:<7}{int(conf[c].sum()):>5}{100*mean[c]:>8.1f}%{100*std[c]:>7.1f}",
              flush=True)

    # structured errors? report the biggest off-diagonal confusions
    off = conf.clone(); off.fill_diagonal_(0)
    pairs = [(int(off[i, j]), i, j) for i in range(nc) for j in range(nc) if off[i, j] > 0]
    pairs.sort(reverse=True)
    tot_err = sum(p[0] for p in pairs)
    print(f"\n=== top confusions ({tot_err} errors over {a.shape[0]} seeds) ===", flush=True)
    for n, i, j in pairs[:10]:
        print(f"  true {i:2d} -> pred {j:2d}   {n:3d}  ({100*n/tot_err:.1f}% of all errors)",
              flush=True)

    json.dump({"seeds": SEEDS, "n_classes": nc,
               "per_seed_class_acc": per_seed_cls,
               "class_mean": mean.tolist(), "class_std": std.tolist(),
               "confusion_summed": conf.tolist()}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
