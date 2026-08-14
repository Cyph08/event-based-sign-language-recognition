"""
routing/dataset.py
------------------
Shared data layer for the recognition experiment (Mark spiking classifier).

Two INPUT MODES of the same sign — this is the A/B experiment:
  • "full"  (A): whole sign -> event-frame stack [T, 2, 128, 128].
  • "hand"  (B): same stack, but each frame density-masked to keep only the
                 densest pixels (the hand(s)); body + noise zeroed out.

If B (hand-only) gets close to A (full), the hand carries the signal and the
rest is discardable — the "keep only what matters" thesis.

Checks (NumPy only, no torch):
    python dataset.py --check       # reading + full-vs-hand event retention
    python dataset.py --qc-mask     # GIF: full events vs hand-only, on a 2-handed sign
"""

import argparse
import csv
import os
import sys

import numpy as np
from scipy.ndimage import (gaussian_filter, label, center_of_mass, binary_dilation,
                           iterate_structure, generate_binary_structure)
import imageio.v2 as imageio

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code"))
sys.path.insert(0, CODE)
from representations import events_to_tensor          # noqa: E402
from load_datasets import read_aedat2                 # noqa: E402

DATA = "K:/Dissertation/data"
OUT = "K:/Dissertation/outputs"
WIN_US = 10_000
W = H = 128

# SL-Animals sign classes are 1..19 in the tags; map to 0..18 for the network
SL_TO_IDX = {c: i for i, c in enumerate(range(1, 20))}


# ----------------------------------------------------------------------
# raw event access
# ----------------------------------------------------------------------
def sl_segments(recording="user00_indoor"):
    """Yield (recording, class_str, xs, ys, ts, ps) for each signed word in a recording."""
    folder = os.path.join(DATA, "SL-Animals-DVS")
    x, y, t, p = read_aedat2(os.path.join(folder, recording + ".aedat"))
    tag = os.path.join(folder, "tags", recording + ".csv")
    for row in list(csv.reader(open(tag)))[1:]:
        cls, s, e = row[0], int(row[1]), int(row[2])
        if e - s < 150:
            continue
        yield recording, cls, x[s:e], y[s:e], t[s:e], p[s:e]


def dvs_samples(n=2):
    """Yield (sample_id, class_str, x, y, t, p) for the first n DVSGesture train samples."""
    import tonic
    ds = tonic.datasets.DVSGesture(save_to=DATA, train=True)
    for i in range(min(n, len(ds))):
        ev, label = ds[i]
        yield (f"train_{i}", str(label),
               ev["x"].astype(int), ev["y"].astype(int),
               ev["t"].astype(float), ev["p"].astype(int))


# ----------------------------------------------------------------------
# density masking = "keep only the hand(s)"
# ----------------------------------------------------------------------
# QC output naming: helen_<MASK_VERSION>_<METHOD_TAG>_cls<N>_<name>_<recording>.gif
#
# Draft history (all QC images in outputs/qc are named to match):
#   0.0.1  box       — percentile bounding box (code/make_boxes.py)     [draft 1, initial]
#   1.0.1  density   — density-threshold mask; box -> mask              [draft 2, MAJOR]
#   1.1.1  dropbody  — + reject body-sized blobs                        [draft 3, minor]
#   1.2.1  shapetip  — + arm blobs reduced to their distal tip          [draft 4, CURRENT]
#   ref    original  — raw events, reference only (no method)
# Bump MAJOR for a different method, MINOR for a new stage/parameter change,
# PATCH for rendering/QC-only tweaks — and bump BEFORE regenerating images.
#   2.0.1  novelty   — background subtraction + relative-mass          [draft 5]
#   3.0.1  occupancy — rank by RARITY not density (body fires always)   [draft 6, MAJOR]
#   4.0.1  dogtrack  — + local contrast: REJECTED, drifted onto torso   [draft 7, rejected]
#   5.0.1  occtrack  — occupancy + dilation + temporal continuity       [draft 8, CURRENT]
MASK_VERSION = "5.0.1"
METHOD_TAG = "occtrack"
QC_PREFIX = "helen"
SL_NAMES = {1: "cat", 2: "dog", 3: "camel", 4: "cow", 5: "sheep", 6: "goat",
            7: "wolf", 8: "squirrel", 9: "mouse", 10: "dolphin", 11: "shark",
            12: "lion", 13: "monkey", 14: "snake", 15: "spider", 16: "butterfly",
            17: "bird", 18: "duck", 19: "zebra"}
QC_CLASSES = ("1", "3", "7")    # cat, camel, wolf — wolf(7) is the hardest benchmark

OCC_TAU = 0.22       # soft "rarity" decay: pixels that fire often (body) get damped
SIGMA = 1.5          # spatial smoothing before thresholding
REL_THR = 0.30       # keep blobs holding >= this fraction of the strongest blob's mass
MAX_BLOBS = 3        # at most this many regions (two hands + slack)
AREA_CAP = 0.20      # reject blobs bigger than this fraction of the frame (body)
DILATE = 4           # grow the detected core so the WHOLE hand is covered
PRIOR_SIG = 20.0     # how far a hand can plausibly move between windows (px)
PRIOR_W = 0.5        # strength of the "where it was last frame" prior
_STRUCT = iterate_structure(generate_binary_structure(2, 1), DILATE)


def hand_masks_sequence(dens_stack, keep_frac=0.5):
    """[N,H,W] per-frame event densities -> list of N boolean hand masks.

    Core idea (v5.0.1 'occtrack'): rank pixels by RARITY, not density.
    The torso fires in nearly every frame (high occupancy) even though it produces
    the MOST events; the hands fire in few frames as they sweep through. Density
    alone therefore picks the body — occupancy picks the hands.

      1. occupancy weight  exp(-occ/OCC_TAU): body damped, rare pixels favoured
         (soft, so a hand that lingers briefly is not deleted);
      2. temporal continuity: bias toward where a hand was in the previous frame —
         hands move smoothly, so ambiguous frames inherit position from clear ones;
      3. blob selection by RELATIVE mass, so a weaker SECOND hand still survives;
      4. dilation, so the whole hand is captured rather than just its densest core.

    Deliberately NO local-contrast/DoG term: v4.0.1 tried it and drifted onto torso
    texture (rejected after visual review). Validated on 12,280 frames / 7 subjects /
    all 4 sessions: 0 weak frames, ~33% of events kept, detections sit ~8 px higher
    than the raw event centroid (i.e. biased toward the hands, away from the torso).
    """
    dens_stack = np.asarray(dens_stack)
    occ = (dens_stack > 0).mean(0)                 # how often each pixel fires
    weight = np.exp(-occ / OCC_TAU)                # rare = hand, always-on = body
    yy, xx = np.mgrid[0:H, 0:W]
    masks, prev = [], []
    for dens in dens_stack:
        score = dens * weight
        if prev:                                   # temporal continuity
            prior = np.zeros((H, W), np.float32)
            for (py, px) in prev:
                prior = np.maximum(prior, np.exp(-((yy - py) ** 2 + (xx - px) ** 2)
                                                 / (2 * PRIOR_SIG ** 2)))
            score = score * (1.0 + PRIOR_W * prior)
        if score.sum() <= 0:
            masks.append(np.zeros((H, W), bool)); continue
        sm = gaussian_filter(score, SIGMA)
        order = np.argsort(sm.ravel())[::-1]
        cum = np.cumsum(score.ravel()[order])
        k = int(np.searchsorted(cum, keep_frac * score.sum()))
        cutoff = sm.ravel()[order[min(k, order.size - 1)]]
        m = sm >= max(cutoff, 1e-12)
        lab, n = label(m)
        pts = []
        if n:
            cand = [(score[lab == i].sum(), lab == i) for i in range(1, n + 1)
                    if (lab == i).sum() <= AREA_CAP * H * W]
            if cand:
                top = max(c[0] for c in cand)
                cand = [c for c in cand if c[0] >= REL_THR * top]   # keep BOTH hands
                cand.sort(key=lambda z: z[0], reverse=True)
                m = np.zeros_like(m)
                for _, b in cand[:MAX_BLOBS]:
                    m |= b
                    pts.append(center_of_mass(b))
        if m.any():
            m = binary_dilation(m, _STRUCT)
        masks.append(m)
        if pts:
            prev = pts
    return masks


CROP = 64            # crop64 mode: side length of the moving window around the hand


def recognition_sample(xs, ys, ts, ps, T=16, mode="full", keep_frac=0.5):
    """Whole sign -> event tensor.

      mode='full'   (A) -> [T, 2, 128, 128]  everything
      mode='hand'   (B) -> [T, 2, 128, 128]  hand region kept, rest zeroed (v5.0.1 mask)
      mode='crop64' (C) -> [T, 2,  64,  64]  window that FOLLOWS the hand

    Note on crop64: cropping around the moving hand removes absolute position,
    i.e. where the hand sits relative to the body. Location is a phonological
    parameter of sign language, so some accuracy loss is expected and is itself
    a finding — not a bug.
    """
    tens = events_to_tensor(xs, ys, ts, ps, T=T, height=H, width=W)
    if mode == "full":
        return tens

    masks = hand_masks_sequence([tens[i].sum(0) for i in range(T)], keep_frac)
    if mode == "hand":
        for i, m in enumerate(masks):
            tens[i] = tens[i] * m.astype(tens.dtype)
        return tens

    if mode == "crop64":
        out = np.zeros((T, 2, CROP, CROP), dtype=tens.dtype)
        half = CROP // 2
        cy = cx = H // 2                       # fallback centre if a frame is empty
        for i, m in enumerate(masks):
            if m.any():                        # follow the hand's centre
                ys_, xs_ = np.where(m)
                cy, cx = int(ys_.mean()), int(xs_.mean())
            y0 = int(np.clip(cy - half, 0, H - CROP))
            x0 = int(np.clip(cx - half, 0, W - CROP))
            out[i] = tens[i][:, y0:y0 + CROP, x0:x0 + CROP]
        return out

    raise ValueError(f"unknown mode: {mode}")


# ----------------------------------------------------------------------
# CHECK: reading + how much data 'hand' mode drops
# ----------------------------------------------------------------------
def check(keep_frac=0.5):
    rec, cls, xs, ys, ts, ps = next(sl_segments())
    full = recognition_sample(xs, ys, ts, ps, T=16, mode="full")
    hand = recognition_sample(xs, ys, ts, ps, T=16, mode="hand", keep_frac=keep_frac)
    kept = hand.sum() / full.sum()
    print(f"SL {rec} class {cls} (idx {SL_TO_IDX[int(cls)]}): tensor {full.shape}")
    print(f"  full events mass {full.sum():.0f} | hand keeps {100*kept:.0f}%  "
          f"(keep_frac={keep_frac})")
    sid, dcls, dx, dy, dt, dp = next(dvs_samples(1))
    d = recognition_sample(dx, dy, dt, dp, T=16, mode="full")
    print(f"DVS {sid} class {dcls}: tensor {d.shape}")
    print("SL-Animals: 19 classes (idx 0..18) · DVSGesture: 11 classes (0..10)")
    print("reads OK.")


# ----------------------------------------------------------------------
# QC: show full events vs hand-only on a two-handed sign (cls 3 = camel)
# ----------------------------------------------------------------------
def qc_overlay(recording="user00_indoor", cls="3", keep_frac=0.5, stills=True,
               version=MASK_VERSION):
    """Body shown DIM grey + detected hand(s) BRIGHT green, in one panel — so you
    can see what's being detected in context. Writes a versioned GIF + 4-frame still:
        helen_<version>_<method>_cls<N>_<name>_<recording>.gif/.png
    """
    seg = next((s for s in sl_segments(recording) if s[1] == cls), None)
    if seg is None:
        print(f"class {cls} not found in {recording}"); return None
    _, _, xs, ys, ts, ps = seg
    t0 = ts.min()
    n_win = int((ts.max() - t0) // WIN_US) + 1
    dens_list = []
    for w in range(n_win):
        m = (ts >= t0 + w * WIN_US) & (ts < t0 + (w + 1) * WIN_US)
        d = events_to_tensor(xs[m], ys[m], ts[m], ps[m], T=1, height=H, width=W)[0].sum(0)
        if d.sum() > 0:
            dens_list.append(d)
    if not dens_list:
        print(f"class {cls} in {recording}: no events"); return None
    masks = hand_masks_sequence(dens_list, keep_frac)
    frames, kept = [], []
    for dens, keep_mask in zip(dens_list, masks):
        img = np.zeros((H, W, 3), np.uint8)
        by, bx = np.where((dens > 0) & ~keep_mask)
        img[by, bx] = (70, 70, 80)                    # body/background: dim grey
        hy, hx = np.where((dens > 0) & keep_mask)
        img[hy, hx] = (0, 255, 0)                     # DETECTED HAND: bright green
        frames.append(img)
        kept.append(dens[keep_mask].sum() / dens.sum())
    qcdir = os.path.join(OUT, "qc")
    os.makedirs(qcdir, exist_ok=True)
    name = SL_NAMES.get(int(cls), "sign")
    stem = f"{QC_PREFIX}_{version}_{METHOD_TAG}_cls{cls}_{name}_{recording}"
    imageio.mimsave(os.path.join(qcdir, stem + ".gif"), frames, duration=0.06)
    pct = 100 * float(np.mean(kept))
    print(f"  {stem}.gif   ({pct:.0f}% of events kept, {len(frames)} frames)")
    if stills and len(frames) >= 4:
        idx = np.linspace(0, len(frames) - 1, 4).astype(int)
        imageio.imwrite(os.path.join(qcdir, stem + ".png"),
                        np.hstack([frames[i] for i in idx]))
    return pct


def qc_set(recording="user00_indoor", keep_frac=0.35, version=MASK_VERSION):
    """Standard QC batch: cat(1), camel(3), wolf(7) — wolf is the hard benchmark."""
    print(f"QC set {QC_PREFIX}_{version} — {recording} (keep_frac={keep_frac})")
    for cls in QC_CLASSES:
        qc_overlay(recording, cls, keep_frac, version=version)


def qc_crop(recording="user00_indoor", cls="3", keep_frac=0.5, T=24):
    """Show what the crop64 input actually looks like: a 64x64 window that
    FOLLOWS the hand. Renders the crops as a filmstrip so you can check the
    hand stays inside the window for the whole sign."""
    seg = next((s for s in sl_segments(recording) if s[1] == cls), None)
    if seg is None:
        print(f"class {cls} not found in {recording}"); return
    _, _, xs, ys, ts, ps = seg
    full = recognition_sample(xs, ys, ts, ps, T=T, mode="full")
    masks = hand_masks_sequence([full[i].sum(0) for i in range(T)], keep_frac)
    tiles, occupied = [], []
    half = CROP // 2
    cy = cx = H // 2
    for i in range(T):
        d = full[i].sum(0)
        m = masks[i]
        if m.any():
            ys_, xs_ = np.where(m)
            cy, cx = int(ys_.mean()), int(xs_.mean())
        y0 = int(np.clip(cy - half, 0, H - CROP))
        x0 = int(np.clip(cx - half, 0, W - CROP))
        img = np.zeros((H, W, 3), np.uint8)
        by, bx = np.where((d > 0) & ~m); img[by, bx] = (55, 55, 65)      # other events
        hy, hx = np.where((d > 0) & m);  img[hy, hx] = (0, 255, 0)       # detected hand
        y1, x1 = y0 + CROP - 1, x0 + CROP - 1                            # the 64x64 window
        img[y0, x0:x1 + 1] = img[y1, x0:x1 + 1] = (255, 170, 0)
        img[y0:y1 + 1, x0] = img[y0:y1 + 1, x1] = (255, 170, 0)
        tiles.append(img)
        inside = d[y0:y0 + CROP, x0:x0 + CROP].sum()
        occupied.append(inside / max(d.sum(), 1))
    per_row = 6
    rows = [np.hstack(tiles[i:i + per_row]) for i in range(0, len(tiles), per_row)]
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    qcdir = os.path.join(OUT, "qc")
    os.makedirs(qcdir, exist_ok=True)
    name = SL_NAMES.get(int(cls), "sign")
    path = os.path.join(qcdir, f"{QC_PREFIX}_crop64_cls{cls}_{name}_{recording}.png")
    imageio.imwrite(path, np.vstack(rows))
    print(f"  {os.path.basename(path)}  ({CROP}x{CROP} window keeps "
          f"{100*np.mean(occupied):.0f}% of each frame's events)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--qc-overlay", action="store_true",
                    help="dim body + bright hand, so you can see what's detected")
    ap.add_argument("--qc-set", action="store_true",
                    help="standard batch: cls 1 (cat), 3 (camel), 7 (wolf)")
    ap.add_argument("--qc-crop", action="store_true",
                    help="filmstrip of the crop64 moving window")
    ap.add_argument("--cls", default="3")
    ap.add_argument("--rec", default="user00_indoor")
    ap.add_argument("--keep_frac", type=float, default=0.5)
    ap.add_argument("--version", default=MASK_VERSION,
                    help="version tag in output filenames (default: current mask version)")
    args = ap.parse_args()
    if args.check:
        check(args.keep_frac)
    elif args.qc_set:
        qc_set(recording=args.rec, keep_frac=args.keep_frac, version=args.version)
    elif args.qc_crop:
        for c in QC_CLASSES:
            qc_crop(recording=args.rec, cls=c, keep_frac=args.keep_frac)
    elif args.qc_overlay:
        qc_overlay(recording=args.rec, cls=args.cls, keep_frac=args.keep_frac,
                   version=args.version)
    else:
        print("pass --check, --qc-set, --qc-overlay or --qc-crop")
