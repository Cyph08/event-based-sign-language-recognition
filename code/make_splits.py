"""
make_splits.py
--------------
Freeze the train/val/test splits so every experiment is reproducible.

SL-Animals-DVS: SUBJECT-INDEPENDENT. 59 recordings named userXX_<session>.
We split by *person* (userXX) so no subject appears in two splits — otherwise the
model could memorise a person instead of learning the sign. Default 70/15/15,
seeded. (If the professor wants the paper's 4-fold CV instead, that's a one-liner.)

DVSGesture: reuse the dataset's OFFICIAL train/test split (via tonic). We carve a
small seeded validation slice out of train for early stopping.

Output -> data/splits/sl_animals.json, data/splits/dvsgesture.json
"""

import glob
import json
import os
import random

DATA = "K:/Dissertation/data"
SPLITDIR = os.path.join(DATA, "splits")
SEED = 42
RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def split_sl_animals():
    folder = os.path.join(DATA, "SL-Animals-DVS")
    recs = [os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(folder, "*.aedat")))]
    if not recs:
        print("[SL] no recordings found"); return
    users = sorted({r.split("_")[0] for r in recs})   # userXX
    rng = random.Random(SEED)
    rng.shuffle(users)
    n = len(users)
    n_tr = round(n * RATIOS[0])
    n_va = round(n * RATIOS[1])
    groups = {"train": set(users[:n_tr]),
              "val": set(users[n_tr:n_tr + n_va]),
              "test": set(users[n_tr + n_va:])}
    split = {k: sorted(r for r in recs if r.split("_")[0] in g)
             for k, g in groups.items()}
    split["_meta"] = {"protocol": "subject-independent", "seed": SEED,
                      "ratios": RATIOS, "n_users": n, "unit": "recording name"}
    _write("sl_animals", split)
    # guarantee no subject leaks across splits
    seen = [set(r.split("_")[0] for r in split[k]) for k in ("train", "val", "test")]
    assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2]), \
        "subject leaked across splits!"
    print(f"[SL] users {n} -> train {len(split['train'])} / "
          f"val {len(split['val'])} / test {len(split['test'])} recordings (subject-independent OK)")


def split_dvsgesture():
    try:
        import tonic
        tr = tonic.datasets.DVSGesture(save_to=DATA, train=True)
        te = tonic.datasets.DVSGesture(save_to=DATA, train=False)
    except Exception as ex:
        print(f"[DVS] skipped: {ex}"); return
    n_tr = len(tr)
    idx = list(range(n_tr))
    random.Random(SEED).shuffle(idx)
    n_val = round(n_tr * 0.15)
    split = {"train": sorted(idx[n_val:]),
             "val": sorted(idx[:n_val]),
             "test": list(range(len(te))),
             "_meta": {"protocol": "official train/test; val carved from train",
                       "seed": SEED, "unit": "sample index into tonic DVSGesture(train=True/False)"}}
    _write("dvsgesture", split)
    print(f"[DVS] train {len(split['train'])} / val {len(split['val'])} "
          f"(from official train {n_tr}) / test {len(split['test'])} samples")


def _write(name, split):
    os.makedirs(SPLITDIR, exist_ok=True)
    with open(os.path.join(SPLITDIR, f"{name}.json"), "w") as f:
        json.dump(split, f, indent=2)


if __name__ == "__main__":
    split_sl_animals()
    split_dvsgesture()
