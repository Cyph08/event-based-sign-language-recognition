"""
routing/qc_hand_samples.py
--------------------------
SAMPLES ONLY — generates comparison images so we can pick how to isolate the hand.
Writes nothing but PNGs into outputs/qc/. No dataset, cache, or CSV is touched.

Two candidate methods:
  M1 "tighter threshold" — keep the densest pixels holding keep_frac of events.
  M2 "drop-body"         — same, then split into blobs, REJECT big diffuse blobs
                           (torso) and keep the top-2 dense ones (the hands).

    python qc_hand_samples.py
"""

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, label
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset as D
from representations import events_to_tensor

OUT = os.path.join(D.OUT, "qc")
H = W = 128
SIGMA = 2.0


# ----------------------------------------------------------------------
def _density(frame):
    return frame.sum(0)


def m1_mask(frame, keep_frac):
    """Keep densest pixels holding keep_frac of the events (current method)."""
    dens = _density(frame)
    total = dens.sum()
    if total == 0:
        return np.zeros_like(dens, bool)
    sm = gaussian_filter(dens, SIGMA)
    order = np.argsort(sm.ravel())[::-1]
    cum = np.cumsum(dens.ravel()[order])
    k = int(np.searchsorted(cum, keep_frac * total))
    cutoff = sm.ravel()[order[min(k, order.size - 1)]]
    return sm >= cutoff


def m2_mask(frame, keep_frac=0.5, area_cap=0.12, top_k=2):
    """M1, then drop blobs bigger than area_cap of the frame (body) and keep the
    top_k densest remaining blobs (the hands)."""
    base = m1_mask(frame, keep_frac)
    if not base.any():
        return base
    dens = _density(frame)
    lab, n = label(base)
    if n == 0:
        return base
    max_area = area_cap * H * W
    cand = []
    for i in range(1, n + 1):
        blob = lab == i
        area = blob.sum()
        mass = dens[blob].sum()
        if area > max_area:          # big + diffuse -> torso/background
            continue
        cand.append((mass, blob))
    if not cand:                      # everything was 'big': fall back to densest blob
        masses = [( dens[lab == i].sum(), lab == i) for i in range(1, n + 1)]
        cand = [max(masses, key=lambda z: z[0])]
    cand.sort(key=lambda z: z[0], reverse=True)
    out = np.zeros_like(base)
    for _, blob in cand[:top_k]:
        out |= blob
    return out


# ----------------------------------------------------------------------
def m3_mask(frame, keep_frac=0.35, area_cap=0.12, top_k=2,
            aspect_thr=2.2, tip_frac=0.40):
    """M2 + shape filter + distal tip.
    A blob that is long and thin (aspect > aspect_thr) is an ARM, not a hand:
    keep only its far end — the tip furthest from the body centre, since the
    hand sits at the end of the arm. Compact blobs are kept whole."""
    base = m1_mask(frame, keep_frac)
    if not base.any():
        return base
    dens = _density(frame)
    body_c = np.array(np.argwhere(dens > 0).mean(0))      # (y, x) centre of the person
    lab, n = label(base)
    if n == 0:
        return base
    max_area = area_cap * H * W

    cand = []
    for i in range(1, n + 1):
        blob = lab == i
        pts = np.argwhere(blob)                            # (y, x)
        if len(pts) < 4 or blob.sum() > max_area:
            continue
        c = pts.mean(0)
        u, s, _ = np.linalg.svd(pts - c, full_matrices=False)
        aspect = (s[0] / s[1]) if s[1] > 1e-6 else 999.0   # elongation
        if aspect > aspect_thr:                            # arm-like -> take the far tip
            axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
            proj = (pts - c) @ axis
            # which end is farther from the body centre?
            end_hi = c + axis * proj.max()
            end_lo = c + axis * proj.min()
            far_hi = np.linalg.norm(end_hi - body_c) > np.linalg.norm(end_lo - body_c)
            span = proj.max() - proj.min()
            sel = (proj >= proj.max() - tip_frac * span) if far_hi else \
                  (proj <= proj.min() + tip_frac * span)
            keep_pts = pts[sel]
            nb = np.zeros_like(base)
            nb[keep_pts[:, 0], keep_pts[:, 1]] = True
            blob = nb
        cand.append((dens[blob].sum(), blob))

    if not cand:
        return m2_mask(frame, keep_frac, area_cap, top_k)   # fallback: never go empty
    cand.sort(key=lambda z: z[0], reverse=True)
    out = np.zeros_like(base)
    for _, blob in cand[:top_k]:
        out |= blob
    return out


def busy_frames(cls, recording="user00_indoor", n=3):
    """Pick n well-separated, busy 10 ms frames from a sign."""
    seg = next((s for s in D.sl_segments(recording) if s[1] == cls), None)
    if seg is None:
        return []
    _, _, xs, ys, ts, ps = seg
    t0 = ts.min()
    n_win = int((ts.max() - t0) // D.WIN_US) + 1
    scored = []
    for w in range(n_win):
        m = (ts >= t0 + w * D.WIN_US) & (ts < t0 + (w + 1) * D.WIN_US)
        c = int(m.sum())
        if c >= 300:
            scored.append((c, w, m))
    if not scored:
        return []
    scored.sort(reverse=True, key=lambda z: z[0])
    picked, used = [], []
    for c, w, m in scored:                      # spread out in time
        if all(abs(w - u) > n_win // (n + 2) for u in used):
            picked.append((w, events_to_tensor(xs[m], ys[m], ts[m], ps[m], T=1)[0]))
            used.append(w)
        if len(picked) == n:
            break
    return picked


def panel(dens, mask=None):
    """One 128x128 RGB tile: kept pixels only (green), nothing else drawn."""
    img = np.zeros((H, W, 3), np.uint8)
    if mask is None:
        yy, xx = np.where(dens > 0)
        img[yy, xx] = (200, 200, 200)           # reference: all events, grey
    else:
        yy, xx = np.where((dens > 0) & mask)
        img[yy, xx] = (0, 230, 0)               # kept only, green
    return img


def grid(rows, path, pad=6):
    """rows = list of (label, [tiles]) -> stacked PNG with a label strip."""
    n_r, n_c = len(rows), max(len(t) for _, t in rows)
    strip = 14
    img = np.zeros((n_r * (H + strip + pad), n_c * (W + pad), 3), np.uint8)
    for r, (lab, tiles) in enumerate(rows):
        y0 = r * (H + strip + pad) + strip
        for c, tile in enumerate(tiles):
            x0 = c * (W + pad)
            img[y0:y0 + H, x0:x0 + W] = tile
        # crude text bar: brightness-code the label into the strip is unreadable;
        # instead print labels to console in matching order.
        img[y0 - strip:y0 - 2, :] = (35, 35, 35)
    imageio.imwrite(path, img)
    print(f"  -> {path}")
    for r, (lab, _) in enumerate(rows):
        print(f"     row {r+1}: {lab}")


# ----------------------------------------------------------------------
def run_for(cls):
    frames = busy_frames(cls)
    if not frames:
        print(f"class {cls}: no busy frames"); return
    dens = [_density(f) for _, f in frames]
    print(f"\n=== class {cls} — frames {[w for w, _ in frames]} ===")

    grid([("ORIGINAL (all events)", [panel(d) for d in dens])],
         os.path.join(OUT, f"sample_original_cls{cls}.png"))

    rows = []
    for kf in (0.5, 0.35, 0.25, 0.15):
        tiles, pcts = [], []
        for (_, f), d in zip(frames, dens):
            m = m1_mask(f, kf)
            tiles.append(panel(d, m)); pcts.append(100 * d[m].sum() / d.sum())
        rows.append((f"M1 keep_frac={kf}  (kept {np.mean(pcts):.0f}% of events)", tiles))
    print(f"\nM1 (tighter threshold) — class {cls}")
    grid(rows, os.path.join(OUT, f"sample_M1_cls{cls}.png"))

    rows = []
    for cap, kf in ((0.12, 0.5), (0.08, 0.5), (0.12, 0.35), (0.08, 0.35)):
        tiles, pcts = [], []
        for (_, f), d in zip(frames, dens):
            m = m2_mask(f, keep_frac=kf, area_cap=cap)
            tiles.append(panel(d, m)); pcts.append(100 * d[m].sum() / d.sum())
        rows.append((f"M2 drop-body cap={cap} keep={kf}  (kept {np.mean(pcts):.0f}%)", tiles))
    print(f"\nM2 (drop-body blob filter) — class {cls}")
    grid(rows, os.path.join(OUT, f"sample_M2_cls{cls}.png"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for cls in ("3", "1"):        # 3 = two-handed (the flagged one), 1 = simpler
        run_for(cls)
    print("\nSamples written. Nothing else on disk was changed.")
