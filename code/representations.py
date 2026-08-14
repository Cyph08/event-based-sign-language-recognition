"""
representations.py
------------------
Turn a raw event stream into the tensor a (spiking) network eats.

events_to_tensor(x, y, t, p, T) -> float array [T, 2, H, W]
  Chop the sample into T equal time-slices; each slice is a 2-channel image:
  channel 0 = OFF-event counts, channel 1 = ON-event counts, per pixel.
  This is the standard "event-frame stack" used by event-based SNNs / Spiking YOLO.

Kept as a library function (not pre-dumped to disk) so T stays a tunable knob and
the dataloader builds tensors on the fly at train time.
"""

import numpy as np

H = W = 128


def events_to_tensor(x, y, t, p, T=16, height=H, width=W, normalize=True):
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    p = (np.asarray(p) > 0).astype(np.int64)   # ON=1, OFF=0
    out = np.zeros((T, 2, height, width), dtype=np.float32)
    if len(t) == 0:
        return out
    # which time-slice each event falls into (0..T-1)
    t0, t1 = t.min(), t.max()
    span = max(t1 - t0, 1.0)
    bins = np.minimum(((t - t0) / span * T).astype(np.int64), T - 1)
    # clip coords defensively (some sensors emit off-by-one edge pixels)
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    np.add.at(out, (bins, p, y, x), 1.0)       # accumulate counts
    if normalize and out.max() > 0:
        out /= out.max()                        # scale to 0..1
    return out


def demo():
    # 2 events per (slice, polarity) at known spots -> counts must land right.
    T = 4
    xs = np.array([10, 10, 20, 20]); ys = np.array([5, 5, 6, 6])
    ps = np.array([1, 1, 0, 0])
    ts = np.array([0.0, 0.0, 300.0, 300.0])     # first pair slice 0, second slice 3
    out = events_to_tensor(xs, ys, ts, ps, T=T, normalize=False)
    assert out.shape == (T, 2, H, W), out.shape
    assert out[0, 1, 5, 10] == 2, "two ON events at slice0 (5,10)"
    assert out[3, 0, 6, 20] == 2, "two OFF events at slice3 (6,20)"
    assert out.sum() == 4, "no events lost or duplicated"
    print("representations demo OK:", out.shape, "total events", int(out.sum()))


if __name__ == "__main__":
    demo()
