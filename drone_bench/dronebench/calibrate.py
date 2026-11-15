"""Parameter calibration for all trackers on training seeds + held-out evaluation.

Protocol
  * every sequence has `pre` pre-roll frames: all algorithms see them (background models learn, tracks
    start) but metrics are computed on frames pre..end only;
  * calibration minimises mean GOSPA (c=10 px, p=2, alpha=2) over TRAINING seeds, by a random search
    followed by a shrinking local search around the incumbent (the defaults are always trial 0);
  * IE-PHPMHT priors (alpha, beta, gamma, p_s, p_b) come from the empirical-Bayes calibration on a
    separate labelled CALIBRATION seed, recomputed whenever a front-end parameter changes;
  * the chosen configurations are compared on TEST seeds never used above.
Latency: detection pipelines = detector + tracker per frame; IE-PHPMHT = front end + filter per frame
(single thread, CPU, frames already in memory).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
import pandas as pd

from .detector import DetectorConfig, detect_sequence
from .metrics import detector_stats, evaluate
from .scenario import ScenarioConfig, simulate
from .tbd import TBDConfig, eb_calibrate, effective_priors, run_tbd
from .trackers import TrackerConfig, make_tracker

BAD = 1e3


# ============================================================== generic search
@dataclass
class Param:
    name: str                          # "detector.threshold", "tracker.P_D", "tbd.B", ...
    lo: float = 0.0
    hi: float = 1.0
    log: bool = False
    integer: bool = False
    choices: tuple | None = None
    step: float | None = None          # round to a grid (e.g. detector threshold, to reuse detections)

    def decode(self, u):
        if self.choices is not None:
            return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]
        if self.log:
            v = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        else:
            v = self.lo + u * (self.hi - self.lo)
        if self.step:
            v = round(v / self.step) * self.step
        if self.integer:
            return int(round(v))
        return float(v)

    def encode(self, v):
        if self.choices is not None:
            return (self.choices.index(v) + 0.5) / len(self.choices)
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)


def search(objective, space, x0, n_random=20, n_local=20, seed=0, step0=0.2, verbose=True, label=""):
    """Minimise objective(dict name -> value). Returns best dict, best score, history DataFrame."""
    rng = np.random.default_rng(seed)
    hist = []
    t_start = time.perf_counter()

    def ev(x, phase):
        t0 = time.perf_counter()
        try:
            y = float(objective(x))
        except Exception as e:                       # a broken configuration is just a bad one
            y = BAD
            if verbose:
                print(f"   ! {type(e).__name__}: {e}")
        if not np.isfinite(y):
            y = BAD
        hist.append(dict(x, score=y, phase=phase, sec=time.perf_counter() - t0))
        return y

    clip_u = lambda u: min(max(u, 0.0), 1.0 - 1e-9)
    u0 = {p.name: clip_u(p.encode(x0[p.name])) for p in space}
    best_u, best_x = u0, {p.name: p.decode(u0[p.name]) for p in space}
    best = ev(best_x, "default")
    if verbose:
        print(f"[{label}] default: {best:.3f}")
    for i in range(n_random):
        u = {p.name: rng.random() for p in space}
        x = {p.name: p.decode(u[p.name]) for p in space}
        y = ev(x, "random")
        if y < best:
            best, best_u, best_x = y, u, x
            if verbose:
                print(f"[{label}] random {i + 1}/{n_random}: {y:.3f}  {_fmt(x)}")
    step, fails = step0, 0
    for i in range(n_local):
        u = {}
        for p in space:
            if p.choices is not None:
                u[p.name] = rng.random() if rng.random() < step else best_u[p.name]
            else:
                u[p.name] = clip_u(best_u[p.name] + rng.normal(0, step) * (rng.random() < 0.6))
        x = {p.name: p.decode(u[p.name]) for p in space}
        if x == best_x:
            continue
        y = ev(x, "local")
        if y < best:
            best, best_u, best_x, fails = y, u, x, 0
            if verbose:
                print(f"[{label}] local {i + 1}/{n_local}: {y:.3f}  {_fmt(x)}")
        else:
            fails += 1
            if fails >= 4:
                step, fails = max(step * 0.7, 0.03), 0
    if verbose:
        print(f"[{label}] best {best:.3f} after {len(hist)} trials, {time.perf_counter() - t_start:.0f}s")
    return best_x, best, pd.DataFrame(hist)


def _fmt(x):
    return ", ".join(f"{k.split('.')[-1]}={v:.3g}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
                     for k, v in x.items())


# ============================================================== benchmark harness
class Bench:
    """Sequences of one scenario for a set of seeds, with cached detections per detector config."""

    def __init__(self, scn: ScenarioConfig, seeds, pre=60, c=10.0, thr=5.0):
        self.scn, self.seeds, self.pre, self.c, self.thr = scn, list(seeds), pre, c, thr
        self.seqs = {s: simulate(scn, s) for s in self.seeds}
        self._det = {}

    def truth(self, s):
        tr = self.seqs[s].truth[self.pre:]
        return [t["ids"] for t in tr], [t["pos"] for t in tr]

    def detections(self, s, det: DetectorConfig):
        key = (s, json.dumps(asdict(det), sort_keys=True))
        if key not in self._det:
            Z, _, lat = detect_sequence(self.seqs[s].frames, det)
            self._det[key] = (Z, lat)
        return self._det[key]

    def score(self, s, ids, pos):
        g_ids, g_pos = self.truth(s)
        summ, frame = evaluate(g_ids, g_pos, ids[self.pre:], pos[self.pre:], c=self.c, thr=self.thr)
        return summ, frame

    # ---- one run of one algorithm on one seed
    def run_detection(self, s, det: DetectorConfig, trk: TrackerConfig):
        seq = self.seqs[s]
        Z, dlat = self.detections(s, det)
        lam = float(np.mean([z.shape[1] for z in Z])) if trk.lambda_c == "auto" else float(trk.lambda_c)
        tr = make_tracker(trk, seq.cfg.width, seq.cfg.height, seq.K, lam)
        ids, pos, lat = [], [], []
        for k in range(seq.K):
            t0 = time.perf_counter()
            i, x = tr.step(Z[k])
            lat.append(time.perf_counter() - t0)
            ids.append(np.asarray(i))
            pos.append(np.asarray(x).reshape(-1, 2))
        summ, frame = self.score(s, ids, pos)
        pd_, fa = detector_stats([t["pos"] for t in seq.truth[self.pre:]], Z[self.pre:])
        sl = slice(self.pre, None)
        return dict(summary=summ, frame=frame, ids=ids, pos=pos, det_ms=dlat[sl] * 1e3,
                    trk_ms=np.array(lat)[sl] * 1e3, det_pd=np.nanmean(pd_), det_fa=fa.mean())

    def run_tbd(self, s, cfg: TBDConfig, cal):
        r = run_tbd(self.seqs[s], cfg, cal)
        summ, frame = self.score(s, r["ids"], r["pos"])
        sl = slice(self.pre, None)
        return dict(summary=summ, frame=frame, ids=r["ids"], pos=r["pos"], det_ms=r["frontend"][sl] * 1e3,
                    trk_ms=r["filter"][sl] * 1e3, n_bern=r["n_bern"][sl])


# ============================================================== search spaces
DET_SPACE = [Param("detector.threshold", 3.0, 5.5, step=0.25)]
RFS_SPACE = DET_SPACE + [
    Param("tracker.P_D", 0.4, 0.95),
    Param("tracker.sigma_v", 0.15, 2.0, log=True),
    Param("tracker.sigma_r", 0.4, 2.5, log=True),
    Param("tracker.lambda_b", 0.02, 0.6, log=True),
    Param("tracker.p_birth", 1e-3, 0.08, log=True),
    Param("tracker.P_S", 0.9, 0.995),
    Param("tracker.birth_vel_std", 0.5, 4.0),
]
GNN_SPACE = DET_SPACE + [
    Param("tracker.sigma_v", 0.15, 2.0, log=True),
    Param("tracker.sigma_r", 0.4, 2.5, log=True),
    Param("tracker.P_D", 0.4, 0.95),
    Param("tracker.confirm_m", choices=(2, 3, 4)),
    Param("tracker.confirm_n", choices=(3, 4, 5, 6)),
    Param("tracker.max_misses", choices=(1, 2, 3, 4, 6, 8)),
]
TBD_FRONTEND = ("B", "sigma_cell", "n_warm", "thr", "K", "a_px", "a_cell", "gain", "offset")
TBD_SPACE = [
    Param("tbd.B", choices=(1, 2)),
    Param("tbd.sigma_cell", 0.5, 2.0, log=True),
    Param("tbd.thr", 3.5, 6.5, step=0.25),
    Param("tbd.n_warm", choices=(20, 40, 60)),
    Param("tbd.a_px", 0.002, 0.03, log=True),
    Param("tbd.omega", 0.04, 0.4, log=True),
    Param("tbd.q_cv_px", 0.005, 0.5, log=True),
    Param("tbd.q_ca_px", 1e-4, 0.05, log=True),
    Param("tbd.p_stay", 0.8, 0.98),
    Param("tbd.t_conf", 0.2, 0.999),
    Param("tbd.birth_vel_std_px", 0.5, 4.0),
    Param("tbd.p_b_scale", 0.2, 30.0, log=True),
    Param("tbd.gamma_mix", 0.0, 2.5),
    Param("tbd.p_s", 0.8, 0.999),
]


def _apply(det: DetectorConfig, trk, x):
    dk = {k.split(".", 1)[1]: v for k, v in x.items() if k.startswith("detector.")}
    tk = {k.split(".", 1)[1]: v for k, v in x.items() if k.startswith("tracker.")}
    return replace(det, **dk), (replace(trk, **tk) if trk is not None else None)


def _tbd_apply(cfg: TBDConfig, x):
    return replace(cfg, **{k.split(".", 1)[1]: v for k, v in x.items() if k.startswith("tbd.")})


def _x0(space, det=None, trk=None, tbd=None):
    out = {}
    for p in space:
        where, attr = p.name.split(".", 1)
        obj = dict(detector=det, tracker=trk, tbd=tbd)[where]
        v = getattr(obj, attr)
        if v is None and attr == "p_s":
            v = 0.99
        out[p.name] = v
    return out


# ============================================================== calibration drivers
def calibrate_detection(bench: Bench, det: DetectorConfig, trk: TrackerConfig, space=None, n_random=20,
                        n_local=20, seed=0, verbose=True):
    space = space or (GNN_SPACE if trk.algo == "GNN" else RFS_SPACE)

    def f(x):
        d, t = _apply(det, trk, x)
        return np.mean([bench.run_detection(s, d, t)["summary"]["GOSPA"] for s in bench.seeds])

    best, score, hist = search(f, space, _x0(space, det, trk), n_random, n_local, seed, verbose=verbose,
                               label=trk.label())
    d, t = _apply(det, trk, best)
    return d, t, score, hist


class TBDCalibrator:
    """Empirical-Bayes priors cached per front-end configuration (calibration seed only)."""

    def __init__(self, scn: ScenarioConfig, calib_seed=100):
        self.seq = simulate(scn, calib_seed)
        self.seed = calib_seed
        self.cache = {}

    def __call__(self, cfg: TBDConfig):
        key = tuple(getattr(cfg, k) for k in TBD_FRONTEND)
        if key not in self.cache:
            self.cache[key] = eb_calibrate(self.seq, cfg, seed=self.seed)[0]
        return self.cache[key]


def calibrate_tbd(bench: Bench, cfg: TBDConfig, calibrator: TBDCalibrator, space=None, n_random=25,
                  n_local=25, seed=0, verbose=True):
    space = space or TBD_SPACE
    space = [replace(p, choices=tuple(c for c in p.choices if c <= bench.pre)) if p.name == "tbd.n_warm" else p
             for p in space]                         # no tracking before the end of warm-up

    def f(x):
        c = _tbd_apply(cfg, x)
        cal = calibrator(c)
        return np.mean([bench.run_tbd(s, c, cal)["summary"]["GOSPA"] for s in bench.seeds])

    best, score, hist = search(f, space, _x0(space, tbd=cfg), n_random, n_local, seed, verbose=verbose,
                               label="IE-PHPMHT")
    c = _tbd_apply(cfg, best)
    return c, calibrator(c), score, hist


# ============================================================== held-out comparison
def compare(bench: Bench, det_trackers: dict, tbd: dict | None = None, verbose=True):
    """det_trackers: name -> (DetectorConfig, TrackerConfig); tbd: name -> (TBDConfig, cal).
       Returns (long DataFrame one row per seed x algorithm, dict of runs[name][seed])."""
    rows, runs = [], {}
    items = [(n, "det", v) for n, v in det_trackers.items()] + [(n, "tbd", v) for n, v in (tbd or {}).items()]
    for name, kind, (a, b) in items:
        runs[name] = {}
        for s in bench.seeds:
            r = bench.run_detection(s, a, b) if kind == "det" else bench.run_tbd(s, a, b)
            runs[name][s] = r
            tot = r["det_ms"] + r["trk_ms"]
            row = dict(algorithm=name, seed=s, **r["summary"], stage1_ms=r["det_ms"].mean(),
                       stage2_ms=r["trk_ms"].mean(), lat_mean_ms=tot.mean(), lat_p50_ms=np.percentile(tot, 50),
                       lat_p95_ms=np.percentile(tot, 95), lat_p99_ms=np.percentile(tot, 99), lat_max_ms=tot.max())
            rows.append(row)
        if verbose:
            d = pd.DataFrame([r for r in rows if r["algorithm"] == name])
            print(f"{name:22s} GOSPA={d.GOSPA.mean():6.2f}  IDF1={d.IDF1.mean():.2f}  MOTA={d.MOTA.mean():+.2f}  "
                  f"track_recall={d.track_recall.mean():.2f}  latency={d.lat_mean_ms.mean():6.1f} ms/frame")
    return pd.DataFrame(rows), runs


def summary_table(df, cols=("GOSPA", "GOSPA_loc", "GOSPA_missed", "GOSPA_false", "OSPA2", "IDF1", "MOTA", "MOTP",
                            "track_recall", "track_precision", "MT", "ML", "ID_switches", "lat_mean_ms", "lat_p95_ms",
                            "stage1_ms", "stage2_ms")):
    cols = [c for c in cols if c in df.columns]
    g = df.groupby("algorithm", sort=False)[cols]
    m, s = g.mean(), g.std()
    return m, s


# ============================================================== persistence
def to_jsonable(det=None, trk=None, tbd=None, cal=None, score=None):
    out = {}
    if det is not None:
        out["detector"] = asdict(det)
    if trk is not None:
        out["tracker"] = asdict(trk)
    if tbd is not None:
        out["tbd"] = asdict(tbd)
    if cal is not None:
        out["cal"] = {k: float(v) for k, v in cal.items()}
    if score is not None:
        out["train_gospa"] = float(score)
    return out


def _mk(cls, d):
    names = {f.name for f in fields(cls)}
    d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return cls(**d)


def from_jsonable(d):
    det = _mk(DetectorConfig, d["detector"]) if "detector" in d else None
    trk = _mk(TrackerConfig, d["tracker"]) if "tracker" in d else None
    tbd = _mk(TBDConfig, d["tbd"]) if "tbd" in d else None
    return det, trk, tbd, d.get("cal"), d.get("train_gospa")


def save_results(path, results: dict, meta: dict | None = None):
    with open(path, "w") as f:
        json.dump(dict(meta=meta or {}, results=results), f, indent=1, default=float)


def load_results(path):
    with open(path) as f:
        return json.load(f)


# ============================================================== bridge to the viz module
def to_experiment(bench: Bench, seed, runs: dict, det: DetectorConfig, cfgs: dict | None = None):
    """Pack the runs of one seed (from compare) into a runner.Experiment so viz.* can draw them.
       Latency per frame = stage 1 + stage 2 (NaN during the pre-roll)."""
    from .runner import Experiment, TrackRun
    seq = bench.seqs[seed]
    Z, dlat = bench.detections(seed, det)
    pd_, fa = detector_stats([t["pos"] for t in seq.truth], Z)
    trs = {}
    for name, per_seed in runs.items():
        r = per_seed[seed]
        lat = np.full(seq.K, np.nan)
        lat[bench.pre:] = (r["det_ms"] + r["trk_ms"]) / 1e3
        ids = [np.zeros(0, int)] * bench.pre + list(r["ids"][bench.pre:])
        pos = [np.zeros((0, 2))] * bench.pre + list(r["pos"][bench.pre:])
        nh = np.array([len(i) for i in ids])
        g_ids = [t["ids"] for t in seq.truth]
        g_pos = [t["pos"] for t in seq.truth]
        summ, frame = evaluate(g_ids, g_pos, ids, pos, c=bench.c, thr=bench.thr)
        trs[name] = TrackRun(name, (cfgs or {}).get(name), ids, pos, lat, nh, summary=r["summary"], frame_metrics=frame)
    lam = float(np.mean([z.shape[1] for z in Z]))
    return Experiment(seq, det, Z, dlat, pd_, fa, lam, trs)
