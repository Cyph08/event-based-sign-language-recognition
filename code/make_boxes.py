"""
make_boxes.py
-------------
Auto-generate the training labels the datasets don't ship with:
ONE "active-region" bounding box per 10 ms window, around the moving hand(s),
plus a light track (box-centre trajectory + velocity).

This is NOT a spiking network. It's plain event-density geometry:
  take all events in a 10 ms slice -> box the central 70% of the event
  cloud (15th..85th percentile of x and y, robust to scattered noise the
  way a median is) -> add margin -> EMA-smooth across slices.
(A per-pixel blob-labelling approach was tried first and failed: at 10 ms
most pixels fire <=1 time, so blobs evaporate. Percentiles on the raw
cloud don't care about per-pixel density.)
Those boxes later become the targets a convolutional Spiking YOLO learns.

Outputs (per dataset) -> K:/Dissertation/data/boxes/<dataset>.csv
Columns: recording, class, window_idx, t_start_us,
         x_min, y_min, x_max, y_max, cx, cy, vx, vy, n_events, low_activity

Usage (in the `dissertation` env, run from code/):
    python make_boxes.py --selftest         # sanity check, no data needed
    python make_boxes.py --qc 6             # boxes on a few segments + QC GIFs
    python make_boxes.py                     # full run over both datasets
"""

import argparse
import csv
import glob
import os

import numpy as np
import imageio.v2 as imageio

from load_datasets import read_aedat2          # reuse the AEDAT 2.0 parser
from event_visualizer import events_to_frames  # reuse the frame painter

DATA = "K:/Dissertation/data"
OUT = "K:/Dissertation/outputs"
BOXDIR = os.path.join(DATA, "boxes")
QCDIR = os.path.join(OUT, "qc")

W = H = 128
WIN_US = 10_000        # 10 ms window
PCT = 15               # keep central (100-2*PCT)% of the event cloud -> clips noise tails
MIN_EVENTS = 150       # windows with fewer events inherit the previous box
MARGIN = 3             # px padding around the box
EMA = 0.5              # box-corner smoothing across windows (0..1, higher = snappier)


# ----------------------------------------------------------------------
def _box_from_window(xw, yw):
    """Box the central (100-2*PCT)% of the event cloud. Robust to scattered noise
    because the dense signing hand dominates the coordinate distribution."""
    if len(xw) == 0:
        return None
    x_lo, x_hi = np.percentile(xw, [PCT, 100 - PCT])
    y_lo, y_hi = np.percentile(yw, [PCT, 100 - PCT])
    x_min = max(int(x_lo) - MARGIN, 0)
    y_min = max(int(y_lo) - MARGIN, 0)
    x_max = min(int(x_hi) + MARGIN, W - 1)
    y_max = min(int(y_hi) + MARGIN, H - 1)
    if x_max <= x_min or y_max <= y_min:  # degenerate (all events one spot)
        return None
    return (x_min, y_min, x_max, y_max)


def boxes_for_segment(x, y, t, cls, recording):
    """Slice one segment into 10 ms windows -> list of box/track rows."""
    t0, t1 = t.min(), t.max()
    edges = np.arange(t0, t1 + WIN_US, WIN_US)
    rows = []
    prev = None           # last smoothed box
    prev_c = None         # last centre (cx, cy)
    dt_s = WIN_US / 1e6
    for i, start in enumerate(edges[:-1]):
        m = (t >= start) & (t < start + WIN_US)
        n_ev = int(m.sum())
        low = 0
        box = _box_from_window(x[m], y[m]) if n_ev >= MIN_EVENTS else None
        if box is None:                    # quiet window -> hold previous box
            if prev is None:
                continue                   # nothing seen yet, skip
            box, low = prev, 1
        if prev is not None and not low:   # EMA-smooth corners for stability
            box = tuple(int(round(EMA * b + (1 - EMA) * p))
                        for b, p in zip(box, prev))
        prev = box
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        vx, vy = (0.0, 0.0) if prev_c is None else \
                 ((cx - prev_c[0]) / dt_s, (cy - prev_c[1]) / dt_s)
        prev_c = (cx, cy)
        rows.append([recording, cls, i, int(start), *box,
                     round(cx, 1), round(cy, 1), round(vx, 1), round(vy, 1),
                     n_ev, low])
    return rows


# ----------------------------------------------------------------------
def _draw_box(frame, box, trail):
    """Draw the box (green) + centroid trail (yellow) onto an RGB frame."""
    x0, y0, x1, y1 = box
    frame[y0, x0:x1 + 1] = (0, 255, 0); frame[y1, x0:x1 + 1] = (0, 255, 0)
    frame[y0:y1 + 1, x0] = (0, 255, 0); frame[y0:y1 + 1, x1] = (0, 255, 0)
    for (cx, cy) in trail:
        yy, xx = int(round(cy)), int(round(cx))
        if 0 <= yy < H and 0 <= xx < W:
            frame[yy, xx] = (255, 255, 0)
    return frame


def qc_gif(x, y, t, rows, path):
    """Render a segment with its boxes drawn, to eyeball quality."""
    frames = events_to_frames(x, y, t, np.zeros_like(x), W, H, WIN_US)
    trail = []
    out = []
    by_idx = {r[2]: r for r in rows}
    for i, fr in enumerate(frames):
        r = by_idx.get(i)
        if r is not None:
            box = tuple(r[4:8]); trail.append((r[8], r[9]))
            fr = _draw_box(fr.copy(), box, trail[-25:])
        out.append(fr)
    imageio.mimsave(path, out, duration=0.05)
    print(f"  QC GIF -> {path}")


# ----------------------------------------------------------------------
def run_sl_animals(qc=0):
    folder = os.path.join(DATA, "SL-Animals-DVS")
    aedats = sorted(glob.glob(os.path.join(folder, "*.aedat")))
    if not aedats:
        print("[SL-Animals] no .aedat found, skipping"); return
    all_rows = []
    qc_done = 0
    for path in aedats:
        rec = os.path.splitext(os.path.basename(path))[0]
        x, y, t, p = read_aedat2(path)
        tag = os.path.join(folder, "tags", rec + ".csv")
        if not os.path.exists(tag):
            print(f"  {rec}: no tag file, skipping"); continue
        with open(tag) as f:
            segs = list(csv.reader(f))[1:]     # skip header
        for cls, s, e in ((r[0], int(r[1]), int(r[2])) for r in segs):
            xs, ys, ts = x[s:e], y[s:e], t[s:e]
            if len(ts) < MIN_EVENTS:
                continue
            rows = boxes_for_segment(xs, ys, ts, cls, rec)
            all_rows += rows
            if qc and qc_done < qc:
                os.makedirs(QCDIR, exist_ok=True)
                qc_gif(xs, ys, ts, rows,
                       os.path.join(QCDIR, f"sl_{rec}_cls{cls}.gif"))
                qc_done += 1
        print(f"  {rec}: {len(segs)} signs boxed")
    _write_csv("sl_animals", all_rows)


def run_dvsgesture(qc=0):
    try:
        import tonic
        ds = tonic.datasets.DVSGesture(save_to=DATA, train=True)
    except Exception as ex:
        print(f"[DVSGesture] skipped: {ex}"); return
    all_rows = []
    n = len(ds) if qc == 0 else min(qc, len(ds))
    for i in range(n):
        ev, label = ds[i]
        x, y, t = ev["x"].astype(int), ev["y"].astype(int), ev["t"].astype(float)
        rows = boxes_for_segment(x, y, t, str(label), f"train_{i}")
        all_rows += rows
        if qc and i < qc:
            os.makedirs(QCDIR, exist_ok=True)
            qc_gif(x, y, t, rows, os.path.join(QCDIR, f"dvs_{i}_cls{label}.gif"))
    _write_csv("dvsgesture", all_rows)


def _write_csv(name, rows):
    os.makedirs(BOXDIR, exist_ok=True)
    path = os.path.join(BOXDIR, f"{name}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "class", "window_idx", "t_start_us",
                    "x_min", "y_min", "x_max", "y_max",
                    "cx", "cy", "vx", "vy", "n_events", "low_activity"])
        w.writerows(rows)
    print(f"[{name}] wrote {len(rows):,} boxes -> {path}")


# ----------------------------------------------------------------------
def selftest():
    """Synthetic blob moving left->right: box must track it, vx must be > 0."""
    xs, ys, ts = [], [], []
    for k in range(10):                       # 10 windows
        cx = 20 + k * 6                        # blob centre moves right
        for _ in range(500):                  # dense blob, well above MIN_EVENTS
            xs.append(cx + np.random.randint(-4, 5))
            ys.append(60 + np.random.randint(-4, 5))
            ts.append(k * WIN_US + np.random.randint(0, WIN_US))
    x = np.clip(np.array(xs), 0, W - 1); y = np.clip(np.array(ys), 0, H - 1)
    t = np.array(ts, dtype=float)
    rows = boxes_for_segment(x, y, t, "test", "selftest")
    assert len(rows) >= 8, f"expected ~10 windows, got {len(rows)}"
    # box centre should sit near the true blob centre
    for r in rows:
        assert 0 <= r[4] < r[6] <= W - 1 and 0 <= r[5] < r[7] <= H - 1, "bad box"
    assert np.mean([r[10] for r in rows[1:]]) > 0, "blob moved right, vx should be +"
    assert rows[-1][8] > rows[0][8], "centre x should increase over time"
    print("selftest OK:", len(rows), "windows, mean vx =",
          round(np.mean([r[10] for r in rows[1:]]), 1), "px/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--qc", type=int, default=0,
                    help="N: only process N segments per dataset + save QC GIFs")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        run_sl_animals(qc=args.qc)
        run_dvsgesture(qc=args.qc)
