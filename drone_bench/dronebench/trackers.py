"""Online (frame-by-frame) adapters around the repo's labeled-RFS trackers.

Wrapped repo code (unchanged, loaded straight from the repo folders):
  * jointglmb_gms_python_fast  - delta-GLMB, joint predict/update, adaptive birth
  * jointlmb_gms_python_fast   - LMB, joint predict/update (GLMB->LMB approximation)

What the adapters add:
  * image-plane model parameters (pixels, frames) instead of the hard-coded demo model
  * a choice of ranked-assignment solver: Murty k-best or Gibbs sampling
  * measurement-driven (adaptive) birth for the LMB, using the same rule the repo's
    GLMB uses (Reuter et al. 2014, eq. 75); the repo LMB only has fixed birth points
  * per-frame MAP estimate extraction, so results are available online and the
    timing is per frame (the repo GLMB's `extract_estimates_recursive` re-runs the
    whole history every frame, i.e. it is a smoother and gets slower as K grows)
  * small robustness patches: empty-measurement gating, larger internal buffers
A simple GNN Kalman tracker is included as a non-RFS baseline (not from the repo).
"""
from __future__ import annotations

import importlib
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats.distributions import chi2

from . import shims

REPO = Path(__file__).resolve().parents[2]
_REPO_MODS = ("run_filter", "utils", "gen_model", "gen_truth", "gen_meas", "plot_results")
SHIM_STATUS = shims.install()


def load_repo_package(subdir: str):
    """Import a repo folder's modules as a fresh, isolated copy (folders reuse module names)."""
    saved = {n: sys.modules.pop(n) for n in _REPO_MODS if n in sys.modules}
    path = str(REPO / subdir)
    sys.path.insert(0, path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ns = SimpleNamespace(run_filter=importlib.import_module("run_filter"),
                                 gen_model=importlib.import_module("gen_model"),
                                 utils=importlib.import_module("utils"))
    finally:
        sys.path.remove(path)
        for n in _REPO_MODS:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
    return ns


@dataclass
class TrackerConfig:
    algo: str = "GLMB"                 # 'GLMB' | 'LMB' | 'GNN'
    assoc: str = "murty"               # 'murty' | 'gibbs'   (RFS trackers)
    sigma_v: float = 0.6               # process noise, px/frame^2 (CV model)
    sigma_r: float = 1.0               # measurement noise, px
    P_D: float = 0.75
    P_S: float = 0.98
    lambda_c: float | str = "auto"     # clutter per frame; 'auto' = mean #detections
    p_birth: float = 0.01              # max birth existence probability per measurement
    lambda_b: float = 0.2              # expected number of births per frame
    birth_pos_std: float = 2.0
    birth_vel_std: float = 2.5
    H_upd: int = 100                   # hypotheses / samples per update
    H_max: int = 200                   # GLMB cap on posterior hypotheses
    hyp_threshold: float = 1e-10
    track_threshold: float = 1e-3      # LMB track pruning on existence
    P_G: float = 0.999                 # gate probability
    # GNN baseline
    confirm_m: int = 3
    confirm_n: int = 5
    max_misses: int = 4

    def label(self):
        return self.algo if self.algo == "GNN" else f"{self.algo}-{self.assoc}"


def _configure_model(gen_model, cfg: TrackerConfig, width, height, lambda_c):
    m = gen_model.model()
    m.sigma_v = cfg.sigma_v
    m.Q = cfg.sigma_v ** 2 * m.B @ m.B.T
    m.P_S, m.Q_S = cfg.P_S, 1 - cfg.P_S
    m.D = np.diag([cfg.sigma_r, cfg.sigma_r])
    m.R = m.D @ m.D.T
    m.P_D, m.Q_D = cfg.P_D, 1 - cfg.P_D
    m.lambda_c = max(float(lambda_c), 0.1)
    m.range_c = np.array([[0.0, width], [0.0, height]])
    m.pdf_c = 1.0 / (width * height)
    B = np.diag([cfg.birth_pos_std, cfg.birth_vel_std, cfg.birth_pos_std, cfg.birth_vel_std])
    m.B_birth = np.repeat(B[None], m.T_birth, 0)
    m.P_birth = np.repeat((B @ B)[None], m.T_birth, 0)
    return m


def _make_gate(utils_gate):
    def gate(z, gamma, model, m, P):
        if z.shape[1] == 0:
            return np.zeros(0, dtype=int)
        return np.asarray(utils_gate(z, gamma, model, m, P)).astype(int)
    return gate


def _solver(name):
    return shims.murty_kbest if name == "murty" else shims.gibbs_jointpredupdt


class _LabelMap:
    def __init__(self):
        self.map = {}

    def __call__(self, lab):
        key = tuple(int(v) for v in np.ravel(lab))
        if key not in self.map:
            self.map[key] = len(self.map)
        return self.map[key]


class GLMBTracker:
    """Repo `jointglmb_gms_python_fast.run_filter.GLMB`, stepped one frame at a time."""

    def __init__(self, cfg: TrackerConfig, width, height, n_frames, lambda_c):
        self.cfg = cfg
        pk = load_repo_package("jointglmb_gms_python_fast")
        rf = self.rf = pk.run_filter
        self.model = _configure_model(pk.gen_model, cfg, width, height, lambda_c)
        f = self.filt = rf.filter(self.model)
        f.H_upd, f.H_max, f.hyp_threshold = cfg.H_upd, cfg.H_max, cfg.hyp_threshold
        f.P_G = cfg.P_G
        f.gamma = chi2.ppf(cfg.P_G, self.model.z_dim)
        f.run_flag = "silence"

        rf.gate_meas_gms_idx = _make_gate(pk.utils.gate_meas_gms_idx)
        solve = _solver(cfg.assoc)
        rf.murty = lambda P0, m: solve(P0, m) if m > 0 else (np.zeros((0, P0.shape[0])), np.zeros(0))

        ah_len = n_frames + 5

        class Target(rf.Target):          # association-history buffer sized to the sequence
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ah = np.zeros(ah_len, dtype=int)
        rf.Target = Target

        p_birth, lambda_b = cfg.p_birth, cfg.lambda_b

        class GLMB(rf.GLMB):
            def apdative_birth(self, assign_prob, meas, model, k):   # repo rule, exposed params
                not_assigned_sum = np.sum(1 - assign_prob) + 1e-12
                self.tt_birth = []
                for idx in range(assign_prob.shape[0]):
                    pb = min(p_birth, (1 - assign_prob[idx]) / not_assigned_sum * lambda_b)
                    self.tt_birth.append(Target(meas.Z[k][:, idx], pb, np.array([k + 1, idx]), model))
                self.tt_birth_steps[k + 1] = None   # not needed online; avoids keeping copies

        self.glmb = GLMB()
        self.meas = SimpleNamespace(Z={}, K=n_frames)
        self.k = 0
        self.labels = _LabelMap()

    def step(self, Z):
        g, k = self.glmb, self.k
        self.meas.Z[k] = Z
        if k == 0:
            g.tt_birth = [self.rf.Target(Z[:, i], self.cfg.p_birth, np.array([0, i]), self.model)
                          for i in range(Z.shape[1])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g.jointpredictupdate(self.model, self.filt, self.meas, k)
            g.prune(self.filt)
            g.cap(self.filt)
        self.meas.Z.pop(k - 1, None)
        self.k += 1
        # MAP cardinality, best hypothesis of that cardinality
        mode = int(np.argmax(g.glmb_update_cdn))
        idx = int(np.argmax(g.glmb_update_w * (g.glmb_update_n == mode)))
        ids, X = [], []
        for i in g.glmb_update_I[idx]:
            tt = g.glmb_update_tt[i]
            ids.append(self.labels(tt.l))
            X.append(np.ravel(tt.m)[[0, 2]])
        return np.array(ids, int), np.array(X).reshape(-1, 2)

    def n_hyp(self):
        return len(self.glmb.glmb_update_w)


class LMBTracker:
    """Repo `jointlmb_gms_python_fast.run_filter.LMB`, with measurement-driven birth."""

    def __init__(self, cfg: TrackerConfig, width, height, n_frames, lambda_c):
        self.cfg = cfg
        pk = load_repo_package("jointlmb_gms_python_fast")
        rf = self.rf = pk.run_filter
        self.model = _configure_model(pk.gen_model, cfg, width, height, lambda_c)
        f = self.filt = rf.filter(self.model)
        f.H_upd, f.track_threshold = cfg.H_upd, cfg.track_threshold
        f.P_G = cfg.P_G
        f.gamma = chi2.ppf(cfg.P_G, self.model.z_dim)
        f.run_flag = "silence"
        rf.gate_meas_gms_idx = _make_gate(pk.utils.gate_meas_gms_idx)

        solve = _solver(cfg.assoc)
        self._last = None

        def recorded(P0, m):
            A, c = solve(P0, m)
            self._last = (A, c, P0.shape[0])
            return A, c
        rf.gibbs_jointpredupdt = recorded

        class Target(rf.Target):          # grow GM buffers on demand (repo fixes them at 200)
            def update_gms(self, model, meas):
                gl = 1 if np.isscalar(self.w) else len(self.w)
                need = gl * (len(self.gatemeas) + 1)
                if self.wg.shape[0] < need:
                    self.wg = np.zeros(need)
                    self.mg = np.zeros((self.x_dim, need))
                    self.Pg = np.zeros((self.x_dim, self.x_dim, need))
                return super().update_gms(model, meas)
        rf.Target = Target

        self.lmb = rf.LMB()
        self.meas = SimpleNamespace(Z={}, K=n_frames)
        self.k = 0
        self.prev_Z = np.zeros((2, 0))
        self.prev_assign = np.zeros(0)
        self.labels = _LabelMap()

    def _set_births(self):
        c, m = self.cfg, self.model
        Zp, a = self.prev_Z, self.prev_assign
        n = Zp.shape[1]
        r = np.minimum(c.p_birth, (1 - a) / (np.sum(1 - a) + 1e-12) * c.lambda_b) if n else np.zeros(0)
        B = np.diag([c.birth_pos_std, c.birth_vel_std, c.birth_pos_std, c.birth_vel_std])
        m.T_birth = n
        m.L_birth = np.ones((n, 1))
        m.r_birth = r.reshape(n, 1)
        m.w_birth = np.ones((n, 1))
        m.m_birth = np.zeros((n, 4, 1))
        if n:
            m.m_birth[:, 0, 0], m.m_birth[:, 2, 0] = Zp[0], Zp[1]
        m.P_birth = np.repeat((B @ B)[None], n, 0)

    def step(self, Z):
        k = self.k
        self.meas.Z[k] = Z
        self._set_births()
        self._last = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.lmb.jointlmbpredictupdate(self.model, self.filt, self.meas, k)
            self.lmb.clean_lmb(self.filt)
            X, N, L = self.lmb.extract_estimates()
        # measurement assignment probabilities -> next frame's births
        a = np.zeros(Z.shape[1])
        if self._last is not None and len(self._last[1]):
            A, cost, nt = self._last
            w = np.exp(-(cost - cost.min()))
            w /= w.sum()
            for h in range(A.shape[0]):
                cols = A[h][A[h] >= 2 * nt].astype(int) - 2 * nt
                a[cols] += w[h]
        self.prev_Z, self.prev_assign = Z, np.clip(a, 0, 1)
        self.meas.Z.pop(k - 1, None)
        self.k += 1
        ids = np.array([self.labels(L[:, i]) for i in range(N)], int)
        return ids, X[[0, 2], :N].T.reshape(-1, 2)

    def n_hyp(self):
        return len(self.lmb.tt_lmb)


class GNNTracker:
    """Baseline: constant-velocity Kalman filters + global nearest neighbour + M/N logic."""

    def __init__(self, cfg: TrackerConfig, width, height, n_frames, lambda_c):
        self.cfg = cfg
        T = 1.0
        self.F = np.array([[1, T, 0, 0], [0, 1, 0, 0], [0, 0, 1, T], [0, 0, 0, 1]], float)
        G = np.array([[T * T / 2, 0], [T, 0], [0, T * T / 2], [0, T]])
        self.Q = cfg.sigma_v ** 2 * G @ G.T
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
        self.R = cfg.sigma_r ** 2 * np.eye(2)
        self.P0 = np.diag([cfg.birth_pos_std, cfg.birth_vel_std, cfg.birth_pos_std, cfg.birth_vel_std]) ** 2
        self.gate = chi2.ppf(cfg.P_G, 2)
        self.tracks = []
        self.next_id = 0

    def step(self, Z):
        c = self.cfg
        for t in self.tracks:
            t["x"] = self.F @ t["x"]
            t["P"] = self.F @ t["P"] @ self.F.T + self.Q
        m, n = Z.shape[1], len(self.tracks)
        used = np.zeros(m, bool)
        if n and m:
            C = np.full((n, m), 1e6)
            for i, t in enumerate(self.tracks):
                S = self.H @ t["P"] @ self.H.T + self.R
                d = Z - (self.H @ t["x"])[:, None]
                md = np.einsum("ij,ij->j", d, np.linalg.solve(S, d))
                C[i, md < self.gate] = md[md < self.gate]
            ri, ci = linear_sum_assignment(C)
            assigned = set()
            for i, j in zip(ri, ci):
                if C[i, j] >= 1e6:
                    continue
                t = self.tracks[i]
                S = self.H @ t["P"] @ self.H.T + self.R
                K = t["P"] @ self.H.T @ np.linalg.inv(S)
                t["x"] = t["x"] + K @ (Z[:, j] - self.H @ t["x"])
                t["P"] = (np.eye(4) - K @ self.H) @ t["P"]
                t["hist"].append(1)
                t["misses"] = 0
                used[j] = True
                assigned.add(i)
            for i, t in enumerate(self.tracks):
                if i not in assigned:
                    t["hist"].append(0)
                    t["misses"] += 1
        else:
            for t in self.tracks:
                t["hist"].append(0)
                t["misses"] += 1
        for t in self.tracks:
            if not t["confirmed"] and sum(t["hist"][-c.confirm_n:]) >= c.confirm_m:
                t["confirmed"] = True
                t["id"] = self.next_id
                self.next_id += 1
        self.tracks = [t for t in self.tracks if
                       (t["confirmed"] and t["misses"] <= c.max_misses) or
                       (not t["confirmed"] and len(t["hist"]) < c.confirm_n + 1 and t["misses"] <= c.confirm_n - c.confirm_m)]
        for j in np.nonzero(~used)[0]:
            self.tracks.append(dict(x=np.array([Z[0, j], 0, Z[1, j], 0.0]), P=self.P0.copy(),
                                    hist=[1], misses=0, confirmed=False, id=-1))
        out = [t for t in self.tracks if t["confirmed"]]
        ids = np.array([t["id"] for t in out], int)
        X = np.array([[t["x"][0], t["x"][2]] for t in out]).reshape(-1, 2)
        return ids, X

    def n_hyp(self):
        return len(self.tracks)


def make_tracker(cfg: TrackerConfig, width, height, n_frames, lambda_c):
    cls = {"GLMB": GLMBTracker, "LMB": LMBTracker, "GNN": GNNTracker}[cfg.algo]
    return cls(cfg, width, height, n_frames, lambda_c)


__all__ = ["TrackerConfig", "make_tracker", "load_repo_package", "SHIM_STATUS", "replace"]
