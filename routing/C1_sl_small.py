"""
routing/C1_sl_small.py
--------------
SL-Animals · SMALL net — can a reduced model keep the accuracy? Use --mode to pick the input.

    python C1_sl_small.py                    # full run
    python C1_sl_small.py --quick            # fast sanity check
    python C1_sl_small.py --mode hand        # switch input (full | hand | crop64)
    python C1_sl_small.py --seed 1           # another seed, for error bars
"""

import argparse
from mark_model import train_and_eval

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "hand", "crop64"])
    ap.add_argument("--keep_frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--net", default=None, help="model: standard|small|base|plif|attn|deep|ghost|wide|combo")
    ap.add_argument("--aug", action="store_true", help="train-time augmentation")
    ap.add_argument("--mix", type=float, default=0.0, help="CutMix probability")
    ap.add_argument("--smooth", type=float, default=0.0, help="label smoothing")
    ap.add_argument("--quick", action="store_true", help="few samples, 3 epochs")
    a = ap.parse_args()
    train_and_eval(code="C1", ds="sl", mode=a.mode, net=a.net or "small",
                   keep_frac=a.keep_frac, T=a.T, seed=a.seed,
                   augment=a.aug, mix_p=a.mix, smooth=a.smooth,
                   epochs=3 if a.quick else a.epochs,
                   limit=6 if a.quick else None)
