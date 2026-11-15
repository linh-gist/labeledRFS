"""Calibrate one algorithm on the training seeds and store the result as JSON.

    python run_calibration.py GLMB-gibbs|LMB-gibbs|GNN|IE-PHPMHT [--trials N] [--out calib]

The scenario/protocol is defined in SETUP below and shared with tbd_vs_rfs_comparison.ipynb, which
loads calib/<algo>.json (or runs this calibration itself if the file is missing).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dronebench import DetectorConfig, ScenarioConfig, TrackerConfig  # noqa: E402
from dronebench.calibrate import (Bench, TBDCalibrator, calibrate_detection, calibrate_tbd,  # noqa: E402
                                  from_jsonable, load_results, save_results, to_jsonable)
from dronebench.tbd import TBDConfig  # noqa: E402

SETUP = dict(snr=2.0, clutter_size_px=(1.0, 5.0), pre=60, n_eval=80, train_seeds=(11, 12, 13), calib_seed=100,
             test_seeds=(1, 2, 3, 4, 5, 6))
BUDGET = {"GLMB-gibbs": (12, 14), "LMB-gibbs": (20, 20), "GNN": (25, 25), "IE-PHPMHT": (30, 30)}


def scenario(setup=SETUP):
    return ScenarioConfig(n_frames=setup["pre"] + setup["n_eval"], snr=setup["snr"],
                          clutter_size_px=tuple(setup["clutter_size_px"]))


def default_configs():
    return {"GLMB-gibbs": (DetectorConfig(), TrackerConfig(algo="GLMB", assoc="gibbs")),
            "LMB-gibbs": (DetectorConfig(), TrackerConfig(algo="LMB", assoc="gibbs")),
            "GNN": (DetectorConfig(), TrackerConfig(algo="GNN")),
            "IE-PHPMHT": (TBDConfig(), None)}


def calibrate_one(algo, n_random=None, n_local=None, setup=SETUP, verbose=True, start=None):
    """start: a previous result dict (JSON 'results') to warm-start from (its best config is trial 0)."""
    scn = scenario(setup)
    bench = Bench(scn, setup["train_seeds"], pre=setup["pre"])
    nr, nl = BUDGET[algo]
    nr = nr if n_random is None else n_random
    nl = nl if n_local is None else n_local
    t0 = time.perf_counter()
    if algo == "IE-PHPMHT":
        x0 = from_jsonable(start)[2] if start else TBDConfig()
        cfg, cal, score, hist = calibrate_tbd(bench, x0, TBDCalibrator(scn, setup["calib_seed"]),
                                              n_random=nr, n_local=nl, verbose=verbose)
        res = to_jsonable(tbd=cfg, cal=cal, score=score)
    else:
        det, trk = default_configs()[algo] if not start else from_jsonable(start)[:2]
        det, trk, score, hist = calibrate_detection(bench, det, trk, n_random=nr, n_local=nl, verbose=verbose)
        res = to_jsonable(det=det, trk=trk, score=score)
    history = json.loads(hist.to_json(orient="records"))
    res["default_train_gospa"] = float(start["default_train_gospa"] if start else hist.score.iloc[0])
    res["history"] = (start["history"] if start else []) + [dict(h, phase=('refine' if start else h['phase'])) for h in history]
    res["n_trials"] = len(res["history"])
    res["seconds"] = (start["seconds"] if start else 0) + time.perf_counter() - t0
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("algo", choices=list(BUDGET))
    ap.add_argument("--random", type=int, default=None)
    ap.add_argument("--local", type=int, default=None)
    ap.add_argument("--refine", action="store_true", help="local search from the stored result")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.algo}.json")
    start = load_results(path)["results"] if a.refine else None
    res = calibrate_one(a.algo, 0 if a.refine and a.random is None else a.random, a.local, start=start)
    save_results(os.path.join(a.out, f"{a.algo}.json"), res, meta=dict(setup=SETUP))
    print("saved", os.path.join(a.out, f"{a.algo}.json"))
