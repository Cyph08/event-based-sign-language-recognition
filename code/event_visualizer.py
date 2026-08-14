"""
event_visualizer.py
-------------------
Turn a raw event stream (x, y, t, polarity) into viewable frames + a GIF.

Core idea: event data is NOT an image. It's a list of tiny "something changed
here" messages. To SEE it, we slice time into short windows and paint all the
events in each window onto a blank canvas -> one frame. Stack frames -> video.

Usage:
    # Demo on synthetic data (no dataset needed):
    python event_visualizer.py --demo

    # On your own Kaggle file (see load_events() for supported formats):
    python event_visualizer.py --file path/to/events.npy --width 128 --height 128
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")           # save images without a screen
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ----------------------------------------------------------------------
# 1) LOAD: get events as four arrays -> x, y, t, p
# ----------------------------------------------------------------------
def load_events(path):
    """
    Return x, y, t, p as numpy arrays.
    Handles the most common Kaggle/event formats. Tell me your exact format
    and I'll tailor this function to it.
    """
    if path.endswith(".npy"):
        arr = np.load(path)                      # expected shape (N, 4): x,y,t,p
        x, y, t, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    elif path.endswith(".csv") or path.endswith(".txt"):
        arr = np.loadtxt(path, delimiter=",")    # same column order
        x, y, t, p = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    else:
        raise ValueError("Unknown format. Send me a sample and I'll add a loader.")
    return x.astype(int), y.astype(int), t.astype(float), p.astype(int)


# ----------------------------------------------------------------------
# 2) SYNTHETIC DEMO: a bright bar sweeping across the sensor
# ----------------------------------------------------------------------
def make_demo_events(width=128, height=128, duration_us=300_000):
    """Fake a moving vertical bar so we have something to visualise."""
    xs, ys, ts, ps = [], [], [], []
    n_steps = 60
    for i in range(n_steps):
        t = i * (duration_us / n_steps)
        bar_x = int((i / n_steps) * width)       # bar moves left -> right
        for yy in range(height):
            # leading edge = brightness up (1), trailing edge = down (0)
            xs.append(bar_x);            ys.append(yy); ts.append(t); ps.append(1)
            xs.append(max(bar_x - 3, 0)); ys.append(yy); ts.append(t); ps.append(0)
    return (np.array(xs), np.array(ys), np.array(ts), np.array(ps))


# ----------------------------------------------------------------------
# 3) SLICE + PAINT: the actual visualisation
# ----------------------------------------------------------------------
def events_to_frames(x, y, t, p, width, height, window_us=30_000):
    """
    Group events into time windows of `window_us` microseconds.
    Each frame: red = brightness up, blue = brightness down, black = nothing.
    """
    t0, t1 = t.min(), t.max()
    edges = np.arange(t0, t1 + window_us, window_us)
    frames = []
    for start in edges[:-1]:
        end = start + window_us
        mask = (t >= start) & (t < end)          # events in this time slice
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        fx, fy, fp = x[mask], y[mask], p[mask]
        up, down = fp == 1, fp == 0
        frame[fy[up],   fx[up]]   = (255, 60, 60)   # brightness increased
        frame[fy[down], fx[down]] = (60, 120, 255)  # brightness decreased
        frames.append(frame)
    return frames


# ----------------------------------------------------------------------
# 4) SAVE: a few PNGs + an animated GIF
# ----------------------------------------------------------------------
def save_outputs(frames, out_prefix="event"):
    # save 3 sample still frames
    picks = np.linspace(0, len(frames) - 1, min(3, len(frames))).astype(int)
    for n, idx in enumerate(picks):
        plt.figure(figsize=(4, 4))
        plt.imshow(frames[idx])
        plt.title(f"Time slice {idx}")
        plt.axis("off")
        plt.savefig(f"{out_prefix}_frame_{n}.png", bbox_inches="tight", dpi=120)
        plt.close()
    # save the animation
    imageio.mimsave(f"{out_prefix}_animation.gif", frames, duration=0.08)
    print(f"Saved {len(frames)} frames -> {out_prefix}_animation.gif "
          f"and {len(picks)} sample PNGs.")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic events")
    ap.add_argument("--file", type=str, help="path to your event file")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--window_us", type=int, default=30_000)
    args = ap.parse_args()

    if args.demo or not args.file:
        print("Running SYNTHETIC demo (a bar sweeping across the sensor)...")
        x, y, t, p = make_demo_events(args.width, args.height)
    else:
        print(f"Loading your file: {args.file}")
        x, y, t, p = load_events(args.file)

    frames = events_to_frames(x, y, t, p, args.width, args.height, args.window_us)
    save_outputs(frames)


if __name__ == "__main__":
    main()