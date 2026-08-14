"""
load_datasets.py
----------------
Phase 1 on REAL data: load one sample from each dissertation dataset,
print its numbers, and render a GIF you can actually watch.

Datasets (expected layout):
    K:/Dissertation/data/DVSGesture/       <- tonic-managed (ibmGestureTrain/Test.tar.gz + extracted)
    K:/Dissertation/data/SL-Animals-DVS/   <- .aedat recordings + tag .csv files

Usage (in the `dissertation` conda env):
    python load_datasets.py
It runs whichever datasets it finds and skips the rest with a message.
"""

import glob
import os

import numpy as np
import imageio.v2 as imageio

from event_visualizer import events_to_frames  # reuse the painter we already have

DATA = "K:/Dissertation/data"
OUT = "K:/Dissertation/outputs"


def describe(name, x, y, t, p, label=None):
    """Print the numbers that matter for the dissertation data chapter."""
    dur_s = (t.max() - t.min()) / 1e6
    print(f"\n=== {name} ===")
    if label is not None:
        print(f"label        : {label}")
    print(f"events       : {len(x):,}")
    print(f"x range      : {x.min()}..{x.max()}   y range: {y.min()}..{y.max()}")
    print(f"duration     : {dur_s:.2f} s")
    print(f"events/sec   : {len(x) / max(dur_s, 1e-9):,.0f}")
    print(f"polarity     : {np.mean(p == 1) * 100:.0f}% ON / {np.mean(p != 1) * 100:.0f}% OFF")


def render_gif(x, y, t, p, out_path, width=128, height=128, window_us=30_000):
    frames = events_to_frames(x, y, t, p, width, height, window_us)
    imageio.mimsave(out_path, frames, duration=0.08)
    print(f"GIF          : {out_path} ({len(frames)} frames)")


# ----------------------------------------------------------------------
# DVS128 Gesture via tonic
# ----------------------------------------------------------------------
def show_dvsgesture():
    try:
        import tonic
        ds = tonic.datasets.DVSGesture(save_to=DATA, train=True)
    except Exception as e:
        print(f"\n[DVSGesture] skipped: {e}")
        return
    events, label = ds[0]
    # tonic gives a structured array with fields x, y, p, t (t in microseconds)
    x, y, t, p = (events["x"].astype(int), events["y"].astype(int),
                  events["t"].astype(float), events["p"].astype(int))
    describe("DVS128 Gesture (train sample 0)", x, y, t, p,
             label=f"{label} = {ds.classes[label]}")
    render_gif(x, y, t, p, os.path.join(OUT, "dvsgesture_sample.gif"))


# ----------------------------------------------------------------------
# SL-Animals-DVS: raw AEDAT 2.0 (jAER) files + tag csv
# ----------------------------------------------------------------------
def read_aedat2(path):
    """AEDAT 2.0: ASCII '#' header lines, then big-endian (uint32 addr, uint32 t) pairs.
    DVS128 address decoding: x=(addr>>1)&0x7F, y=(addr>>8)&0x7F, polarity=addr&1."""
    with open(path, "rb") as f:
        header_end = 0
        while True:
            line = f.readline()
            if not line.startswith(b"#"):
                break
            header_end = f.tell()
        f.seek(header_end)
        raw = np.frombuffer(f.read(), dtype=">u4")
    raw = raw[: (len(raw) // 2) * 2].reshape(-1, 2)
    addr, t = raw[:, 0], raw[:, 1].astype(np.int64)
    x = (addr >> 1) & 0x7F
    y = (addr >> 8) & 0x7F
    p = (addr & 1).astype(int)
    return x.astype(int), y.astype(int), t.astype(float), p


def show_sl_animals():
    folder = os.path.join(DATA, "SL-Animals-DVS")
    aedats = sorted(glob.glob(os.path.join(folder, "**", "*.aedat"), recursive=True))
    if not aedats:
        print(f"\n[SL-Animals-DVS] skipped: no .aedat files in {folder} yet.")
        return
    path = aedats[0]
    x, y, t, p = read_aedat2(path)
    describe(f"SL-Animals-DVS ({os.path.basename(path)})", x, y, t, p)

    # tag file = which event index ranges belong to which of the 19 signs
    # (real layout: tags/<recording>.csv with columns class,startTime_ev,endTime_ev)
    base = os.path.splitext(os.path.basename(path))[0] + ".csv"
    tag = os.path.join(folder, "tags", base)
    if not os.path.exists(tag):
        tag = os.path.splitext(path)[0] + ".csv"
    if os.path.exists(tag):
        import csv
        with open(tag) as f:
            rows = list(csv.reader(f))
        print(f"tag file     : {os.path.basename(tag)}, {len(rows)} rows, "
              f"first rows: {rows[:3]}")
        # try to cut the first tagged sign out of the recording (columns are
        # typically: class, start_event_index, end_event_index)
        try:
            _, s, e = rows[1][0], int(rows[1][1]), int(rows[1][2])
            x, y, t, p = x[s:e], y[s:e], t[s:e], p[s:e]
            print(f"cut sign     : events {s}..{e} ({e - s:,} events)")
        except (ValueError, IndexError):
            print("tag columns differ from expected -- showing whole recording")
    render_gif(x, y, t, p, os.path.join(OUT, "sl_animals_sample.gif"))


if __name__ == "__main__":
    show_dvsgesture()
    show_sl_animals()
