"""IE-PHPMHT track-before-detect (arXiv:2504.20526) on dronebench sequences.

Thin adapter around the refactored `iephpmht` package of the track_before_detect_exp project
(Neural-Glimpse), which runs here on the CPU front end (FrontEndCPU, NumPy) - the same maths as the
project's CUDA kernels. The package is found at
    $IEPHPMHT_ROOT, or ../../track_before_detect_exp next to the labeledRFS clone.

Pipeline per frame (no detection threshold anywhere in the tracking loop):
  front end   raw frame -> per-pixel background model -> one-sided z-residual summed over B x B cells
              -> dispersion-matched counts N + clutter map L0 (Poisson-like), birth statistic Z
  births      top-K local maxima of Z above `thr` that are not near an existing Bernoulli
              (a birth proposal, not a detection: existence decides)
  filter      windowed IE-PHPMHT: IMM (CV, CT+, CT-, CA) Bernoullis, EM on the counts, KLD-merged
              Gamma rate prior (Lemma 1), existence update eq. 47; reported when r > t_conf
Priors (alpha, beta, gamma, p_s, p_b) come from the project's empirical-Bayes calibration on a
labelled calibration sequence (never a test sequence).

Coordinates: dronebench renders a blob centred at x on pixel index x (pixel i is centred at i);
iephpmht treats pixel i as the interval [i, i+1) (centre i + 0.5). The adapter converts both ways.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass, replace

import numpy as np


def _find_root():
    cands = [os.environ.get("IEPHPMHT_ROOT", ""),
             os.path.join(os.path.dirname(__file__), "..", "..", "..", "track_before_detect_exp"),
             os.path.join(os.path.expanduser("~"), "Documents", "gitcloned", "track_before_detect_exp")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "iephpmht")):
            return os.path.abspath(c)
    raise ImportError("iephpmht package not found: set IEPHPMHT_ROOT to the track_before_detect_exp clone")


IEPHPMHT_ROOT = _find_root()
if IEPHPMHT_ROOT not in sys.path:
    sys.path.insert(0, IEPHPMHT_ROOT)
import iephpmht as ip  # noqa: E402

PIX_OFFSET = 0.5          # iephpmht coordinate = dronebench coordinate + 0.5


@dataclass
class TBDConfig:
    # front end
    B: int = 2                       # cell size (px)
    sigma_cell: float = 0.7          # PSF sigma in cells (target sigma ~ B * sigma_cell px)
    n_warm: int = 40                 # frames of fast background learning; no tracking before
    thr: float = 4.5                 # birth-statistic threshold for birth PROPOSALS
    K: int = 50                      # max birth proposals per frame
    a_px: float = 0.01               # background learning rates after warm-up
    a_cell: float = 0.005
    gain: float = 4.0                # dronebench frame (noise sigma 1) -> camera grey levels
    offset: float = 128.0
    # filter
    omega: float = 0.12              # CT turn rate (rad/frame)
    q_cv_px: float = 0.02            # process noise (px^2/frame^3), CV / CT modes
    q_ca_px: float = 2e-3            # CA mode jerk noise
    p_stay: float = 0.93             # IMM mode persistence
    t_conf: float = 0.5              # report Bernoullis with r > t_conf
    birth_pos_std_px: float = 1.0
    birth_vel_std_px: float = 2.0
    p_b_scale: float = 1.0           # multiplies the calibrated p_b
    gamma_mix: float = 0.0           # gamma prior of the 'nothing there' rate, log-interpolated between
                                     # 0: 1/mean rate MLE at a FIXED target-free position (paper)
                                     # 1: 1/mean rate MLE when the position may move +-1 cell (what a
                                     #    Bernoulli that drifted off its target actually sees); >1 extrapolates
    p_s: float | None = None         # survival probability; None = calibrated from the truth stream
    max_em: int = 8

    def fe_kw(self):
        return dict(n_warm=self.n_warm, thr=self.thr, K=self.K, a_px=self.a_px, a_cell=self.a_cell)

    def label(self):
        return "IE-PHPMHT"


class SeqSource:
    """dronebench Sequence -> iephpmht frame source (frames(), truth_at(k), shape)."""

    def __init__(self, seq, gain=4.0, offset=128.0):
        self.seq, self.gain, self.offset = seq, gain, offset
        self.shape = seq.frames.shape[1:]
        self.n_frames = seq.K

    def frame(self, k):
        return np.clip(self.offset + self.gain * self.seq.frames[k], 0, 255).astype(np.float32)

    def frames(self):
        for k in range(self.seq.K):
            yield k, self.frame(k)

    def truth_at(self, k):
        t = self.seq.truth[k]
        return list(t["ids"]), t["pos"] + PIX_OFFSET


def eb_calibrate(seq, cfg: TBDConfig, seed=0):
    """Empirical-Bayes priors on a labelled calibration sequence for this front-end configuration."""
    cal, diag = ip.calibrate_source(SeqSource(seq, cfg.gain, cfg.offset), cfg.B, cfg.sigma_cell,
                                    n_warm=cfg.n_warm, thr=cfg.thr, K=cfg.K, backend="cpu",
                                    fe_kw=dict(a_px=cfg.a_px, a_cell=cfg.a_cell), seed=seed)
    cal["gamma_adapt"] = float(1.0 / diag["off_adapt"].mean())
    return cal, diag


def effective_priors(cfg: TBDConfig, cal: dict):
    """Priors actually used by the filter: calibrated values + the configured adjustments."""
    g0 = cal["gamma"]
    g1 = cal.get("gamma_adapt", g0)
    m = cfg.gamma_mix
    return dict(alpha=cal["alpha"], beta=cal["beta"],
                gamma=float(np.exp((1 - m) * np.log(g0) + m * np.log(g1))),
                p_s=float(cal["p_s"] if cfg.p_s is None else cfg.p_s),
                p_b=float(np.clip(cal["p_b"] * cfg.p_b_scale, 1e-6, 0.5)))


class TBDTracker:
    def __init__(self, cfg: TBDConfig, cal: dict, shape):
        self.cfg, self.shape = cfg, tuple(shape)
        c, B = cfg, cfg.B
        cal = effective_priors(c, cal)
        self.fe = ip.make_frontend(shape, B, c.sigma_cell, backend="cpu", **c.fe_kw())
        sp, sv = (c.birth_pos_std_px / B) ** 2, (c.birth_vel_std_px / B) ** 2
        self.filt = ip.make_filter(cal, B, c.sigma_cell, shape, omega=c.omega, q_cv_px=c.q_cv_px,
                                   q_ca_px=c.q_ca_px, p_stay=c.p_stay, t_conf=c.t_conf, max_em=c.max_em,
                                   birth_P=np.diag([sp, sv, 1e-3, sp, sv, 1e-3]))
        self.k = 0
        self.timing = {}

    def step(self, frame):
        """frame: camera frame (grey levels). Returns (ids, (n,2) positions in dronebench pixels)."""
        t0 = time.perf_counter()
        N, L0 = self.fe.process(frame)
        t1 = time.perf_counter()
        self.k += 1
        if self.k <= self.cfg.n_warm:
            self.timing = dict(frontend=t1 - t0, filter=0.0, n_bern=0)
            return np.zeros(0, int), np.zeros((0, 2))
        cand, _ = self.fe.birth_candidates()
        self.filt.step(N, L0, cand)
        bid, pos, r, lam = self.filt.estimates()
        t2 = time.perf_counter()
        self.timing = dict(frontend=t1 - t0, filter=t2 - t1, n_bern=len(self.filt.r))
        return bid.astype(int), pos * self.cfg.B - PIX_OFFSET

    def n_hyp(self):
        return len(self.filt.r)


def run_tbd(seq, cfg: TBDConfig, cal: dict):
    """Run on a whole sequence -> ids, pos (lists per frame), latency (s, total), frontend/filter (s)."""
    src = SeqSource(seq, cfg.gain, cfg.offset)
    tr = TBDTracker(cfg, cal, src.shape)
    ids, pos, lat, fe_t, fi_t, nb = [], [], [], [], [], []
    for k in range(seq.K):
        F = src.frame(k)                      # frame conversion = camera input, not timed
        t0 = time.perf_counter()
        i, x = tr.step(F)
        lat.append(time.perf_counter() - t0)
        ids.append(i)
        pos.append(x)
        fe_t.append(tr.timing["frontend"])
        fi_t.append(tr.timing["filter"])
        nb.append(tr.timing["n_bern"])
    return dict(ids=ids, pos=pos, latency=np.array(lat), frontend=np.array(fe_t), filter=np.array(fi_t),
                n_bern=np.array(nb))


__all__ = ["TBDConfig", "TBDTracker", "SeqSource", "eb_calibrate", "effective_priors", "run_tbd", "IEPHPMHT_ROOT", "asdict", "replace"]
