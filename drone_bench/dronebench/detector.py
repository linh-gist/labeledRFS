"""Image -> point measurements (the input the RFS trackers expect).

Pipeline per frame:
  optional background subtraction -> high-pass (removes sky gradient and part of
  large blobs) -> matched Gaussian filter -> robust SNR normalisation -> threshold
  + non-max suppression -> optional blob-size gate (scale-normalised LoG) ->
  sub-pixel refinement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class DetectorConfig:
    threshold: float = 4.0             # detection threshold, in std of the filtered image
    mf_sigma: float = 1.0              # matched-filter sigma (px), ~ expected target sigma
    highpass_sigma: float | None = 8.0 # subtract this-blurred image; None = off
    nms_radius: int = 3
    size_gate_sigma: float | None = 2.5  # reject blobs whose best LoG scale exceeds this; None = off
    log_scales: tuple = (0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0)
    bg_subtract: str = "running"       # 'none' | 'running'  (running mean, removes static clutter)
    bg_alpha: float = 0.05
    max_detections: int = 150


class Detector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.bg = None

    def __call__(self, img):
        c = self.cfg
        x = img.astype(np.float32)
        raw = x
        if c.bg_subtract == "running":
            if self.bg is None:
                self.bg = x.copy()
            diff = x - self.bg
            self.bg = (1 - c.bg_alpha) * self.bg + c.bg_alpha * x
            x = diff
        if c.highpass_sigma:
            x = x - ndi.gaussian_filter(x, c.highpass_sigma)
        f = ndi.gaussian_filter(x, c.mf_sigma)
        med = np.median(f)
        noise = 1.4826 * np.median(np.abs(f - med)) + 1e-9
        s = (f - med) / noise
        peaks = (s == ndi.maximum_filter(s, size=2 * c.nms_radius + 1)) & (s > c.threshold)
        ys, xs = np.nonzero(peaks)
        scores = s[ys, xs]
        if c.size_gate_sigma is not None and len(xs):
            # blob scale is measured on the current frame (not the background difference), so residue
            # peaks left by large moving clutter blobs are recognised as part of a large blob
            src = raw - ndi.gaussian_filter(raw, c.highpass_sigma) if c.highpass_sigma else raw
            resp = np.stack([-(sg ** 2) * ndi.gaussian_laplace(src, sg)[ys, xs] for sg in c.log_scales])
            best = np.asarray(c.log_scales)[np.argmax(resp, axis=0)]
            keep = best <= c.size_gate_sigma
            ys, xs, scores = ys[keep], xs[keep], scores[keep]
        if len(xs) > c.max_detections:
            o = np.argsort(-scores)[: c.max_detections]
            ys, xs, scores = ys[o], xs[o], scores[o]
        # sub-pixel refinement: parabolic fit on the SNR map
        H, W = s.shape
        xr = xs.astype(float).copy()
        yr = ys.astype(float).copy()
        inner = (xs > 0) & (xs < W - 1) & (ys > 0) & (ys < H - 1)
        xi, yi = xs[inner], ys[inner]
        l, r0, rr = s[yi, xi - 1], s[yi, xi], s[yi, xi + 1]
        u, d = s[yi - 1, xi], s[yi + 1, xi]
        dx = 0.5 * (l - rr) / np.where(np.abs(l - 2 * r0 + rr) > 1e-9, l - 2 * r0 + rr, 1e-9)
        dy = 0.5 * (u - d) / np.where(np.abs(u - 2 * r0 + d) > 1e-9, u - 2 * r0 + d, 1e-9)
        xr[inner] += np.clip(dx, -0.5, 0.5)
        yr[inner] += np.clip(dy, -0.5, 0.5)
        return np.vstack([xr, yr]), scores


def detect_sequence(frames, cfg: DetectorConfig):
    """Returns list of (2, m) measurement arrays, list of scores, per-frame latency (s)."""
    det = Detector(cfg)
    Z, S, lat = [], [], []
    for img in frames:
        t0 = time.perf_counter()
        z, sc = det(img)
        lat.append(time.perf_counter() - t0)
        Z.append(z)
        S.append(sc)
    return Z, S, np.array(lat)
