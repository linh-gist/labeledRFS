"""Glue: simulate -> detect -> track (timed per frame) -> evaluate; plus parameter sweeps."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .detector import DetectorConfig, detect_sequence
from .metrics import detector_stats, evaluate
from .scenario import ScenarioConfig, Sequence, simulate
from .trackers import TrackerConfig, make_tracker


@dataclass
class TrackRun:
    name: str
    cfg: TrackerConfig
    ids: list
    pos: list
    latency: np.ndarray            # seconds per frame (tracker only)
    n_hyp: np.ndarray              # GLMB hypotheses / LMB tracks / GNN tracks after each frame
    summary: dict = field(default_factory=dict)
    frame_metrics: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class Experiment:
    seq: Sequence
    det_cfg: DetectorConfig
    Z: list
    det_latency: np.ndarray
    det_pd: np.ndarray
    det_fa: np.ndarray
    lambda_c: float
    runs: dict

    def table(self):
        rows = []
        for name, r in self.runs.items():
            row = dict(algorithm=name, **r.summary)
            lat = r.latency[np.isfinite(r.latency)] * 1e3
            if len(lat) == 0:
                lat = np.array([np.nan])
            row.update(lat_mean_ms=np.mean(lat), lat_p50_ms=np.percentile(lat, 50),
                       lat_p95_ms=np.percentile(lat, 95), lat_p99_ms=np.percentile(lat, 99),
                       lat_max_ms=np.max(lat), fps=1e3 / max(np.mean(lat), 1e-9),
                       failed=r.error is not None)
            rows.append(row)
        return pd.DataFrame(rows).set_index("algorithm")


def run_trackers(seq: Sequence, Z, trackers: dict, lambda_c, c=10.0, thr=5.0, verbose=True):
    gt_ids = [t["ids"] for t in seq.truth]
    gt_pos = [t["pos"] for t in seq.truth]
    runs = {}
    for name, tcfg in trackers.items():
        lam = lambda_c if tcfg.lambda_c == "auto" else float(tcfg.lambda_c)
        tr = make_tracker(tcfg, seq.cfg.width, seq.cfg.height, seq.K, lam)
        ids, pos, lat, nh = [], [], [], []
        err = None
        try:
            for k in range(seq.K):
                t0 = time.perf_counter()
                i, x = tr.step(Z[k])
                lat.append(time.perf_counter() - t0)
                ids.append(i)
                pos.append(x)
                nh.append(tr.n_hyp())
        except Exception as e:          # keep other algorithms running; report the failure
            import traceback
            err = f"{type(e).__name__}: {e} (frame {len(ids)})\n" + traceback.format_exc(limit=3)
            for _ in range(len(ids), seq.K):
                ids.append(np.zeros(0, int))
                pos.append(np.zeros((0, 2)))
                lat.append(np.nan)
                nh.append(0)
        r = TrackRun(name, tcfg, ids, pos, np.array(lat), np.array(nh), error=err)
        r.summary, r.frame_metrics = evaluate(gt_ids, gt_pos, ids, pos, c=c, thr=thr)
        runs[name] = r
        if verbose:
            s = r.summary
            msg = (f"{name:14s} GOSPA={s['GOSPA']:6.2f}  IDF1={s['IDF1']:.2f}  MOTA={s['MOTA']:+.2f}  "
                   f"track_recall={s['track_recall']:.2f}  lat={np.nanmean(r.latency) * 1e3:7.1f} ms/frame")
            print(msg if err is None else f"{name:14s} FAILED: {err.splitlines()[0]}")
    return runs


def run_experiment(scn: ScenarioConfig, det: DetectorConfig, trackers: dict, seed=0,
                   c=10.0, thr=5.0, verbose=True) -> Experiment:
    t0 = time.perf_counter()
    seq = simulate(scn, seed)
    Z, _, dlat = detect_sequence(seq.frames, det)
    pd_, fa = detector_stats([t["pos"] for t in seq.truth], Z)
    lam = float(np.mean([z.shape[1] for z in Z]))
    if verbose:
        n_true = np.mean([len(t["ids"]) for t in seq.truth])
        print(f"scene: {seq.K} frames, {len({i for t in seq.truth for i in t['ids']})} drones "
              f"(avg {n_true:.1f} visible) | detector: Pd={np.nanmean(pd_):.2f}, "
              f"false alarms/frame={fa.mean():.1f}, {dlat.mean() * 1e3:.1f} ms/frame "
              f"| sim+detect {time.perf_counter() - t0:.1f}s")
    runs = run_trackers(seq, Z, trackers, lam, c=c, thr=thr, verbose=verbose)
    return Experiment(seq, det, Z, dlat, pd_, fa, lam, runs)


def _set(cfgs, key, val):
    """Apply 'scenario.snr' / 'detector.threshold' / 'tracker.P_D' style keys."""
    scn, det, trk = cfgs
    where, attr = key.split(".", 1)
    if where == "scenario":
        scn = replace(scn, **{attr: val})
    elif where == "detector":
        det = replace(det, **{attr: val})
    elif where == "tracker":
        trk = {n: replace(t, **{attr: val}) for n, t in trk.items()}
    else:
        raise KeyError(key)
    return scn, det, trk


def sweep(scn: ScenarioConfig, det: DetectorConfig, trackers: dict, param: str, values,
          seeds=(0, 1), c=10.0, thr=5.0, progress=True):
    """Monte-Carlo sweep of one parameter. Returns a long DataFrame (one row per value/seed/algo)."""
    rows = []
    total = len(values) * len(seeds)
    i = 0
    for v in values:
        s2, d2, t2 = _set((scn, det, trackers), param, v)
        for sd in seeds:
            i += 1
            if progress:
                print(f"[{i}/{total}] {param}={v} seed={sd}", flush=True)
            ex = run_experiment(s2, d2, t2, seed=sd, c=c, thr=thr, verbose=False)
            tab = ex.table().reset_index()
            tab[param] = v
            tab["seed"] = sd
            tab["det_pd"] = np.nanmean(ex.det_pd)
            tab["det_fa"] = ex.det_fa.mean()
            rows.append(tab)
    return pd.DataFrame(pd.concat(rows, ignore_index=True))
