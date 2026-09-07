"""
make_cv_splits.py
-----------------
4-fold SUBJECT-INDEPENDENT cross-validation splits for SL-Animals-DVS.

WHY. Published SL-Animals results (SLAYER and its reimplementations) report 4-fold
cross-validation. This project used one fixed 41/9/9 subject split, so the headline
figure was not directly comparable and the report had to caveat it. Running the same
protocol as the published work removes the caveat and makes the comparison like-for-like.

PROTOCOL. The 59 signers are partitioned into 4 subject-disjoint folds. For each fold:
    test  = that fold's users            (~15 users)
    val   = a seeded slice of the OTHER three folds' users, held out for early stopping
    train = the remaining users          (~35 users)

Using three folds for train/val rather than two keeps the training-set size close to the
fixed split this project used elsewhere (41 users), so the CV figure differs from the
fixed-split figure because of the PROTOCOL rather than because of a smaller training set.
A separate validation set is retained because every result in this project selects the
model on validation and evaluates test once; dropping it would change the protocol in a
second way and make the comparison harder, not easier.

Reuses the recording-name -> user convention from make_splits.py (userXX_<session>).

Output -> data/splits/sl_animals_cv4.json
"""

import glob
import json
import os
import random

DATA = "K:/Dissertation/data"
SPLITDIR = os.path.join(DATA, "splits")
SEED = 42
K = 4


def main():
    folder = os.path.join(DATA, "SL-Animals-DVS")
    recs = [os.path.splitext(os.path.basename(p))[0]
            for p in sorted(glob.glob(os.path.join(folder, "*.aedat")))]
    if not recs:
        print("no recordings found"); return
    users = sorted({r.split("_")[0] for r in recs})
    rng = random.Random(SEED)
    rng.shuffle(users)

    folds = [users[i::K] for i in range(K)]        # round-robin -> near-equal sizes
    out = {"_meta": {"protocol": "4-fold subject-independent CV", "seed": SEED,
                     "k": K, "n_users": len(users), "unit": "recording name",
                     "note": "test = fold i; val = 8 seeded users from the rest; train = remainder"}}

    N_VAL = 8                                      # users held out for model selection
    for i in range(K):
        te_u = set(folds[i])
        rest = [u for u in users if u not in te_u]
        random.Random(SEED + i).shuffle(rest)       # seeded, so folds are reproducible
        va_u = set(rest[:N_VAL])
        tr_u = set(rest[N_VAL:])
        sel = lambda g: sorted(r for r in recs if r.split("_")[0] in g)
        fold = {"train": sel(tr_u), "val": sel(va_u), "test": sel(te_u)}

        # subject-disjointness is the whole point: assert it, do not assume it
        s = [set(r.split("_")[0] for r in fold[k]) for k in ("train", "val", "test")]
        assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2]), \
            f"fold {i}: subject leaked across splits"
        out[f"fold{i}"] = fold
        print(f"fold{i}: users train {len(tr_u)} / val {len(va_u)} / test {len(te_u)}"
              f"   recordings {len(fold['train'])}/{len(fold['val'])}/{len(fold['test'])}")

    # every user must be tested exactly once across the K folds
    tested = [u for i in range(K) for u in folds[i]]
    assert sorted(tested) == sorted(users), "not every user is tested exactly once"
    print(f"\nall {len(users)} users tested exactly once across {K} folds")

    os.makedirs(SPLITDIR, exist_ok=True)
    p = os.path.join(SPLITDIR, "sl_animals_cv4.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
