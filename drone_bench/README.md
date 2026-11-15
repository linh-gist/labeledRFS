# drone_bench: drone tracking playground for the labeled-RFS trackers

Open `drone_tracking_playground.ipynb` (VS Code or JupyterLab), with this folder as the working directory.

What it does:

* Simulates drone-like targets in a camera image: hover, cruise, accelerate and circle modes, with a variable number of targets.
* Targets are 1–5 px Gaussian blobs with SNR 1–5, in heavy clutter of 1–25 px blobs.
* A detector turns each frame into point measurements.
* The repo's **GLMB** and **LMB** trackers run on them (Murty or Gibbs association), with a GNN baseline for comparison.
* Reports GOSPA / OSPA / OSPA(2) / CLEAR-MOT / IDF1 / track_recall / track_precision, per-frame latency, a frame viewer, animations and Monte-Carlo parameter sweeps.

```
dronebench/
  scenario.py   drone motion, clutter, image rendering
  detector.py   background subtraction, matched filter, NMS, size gate
  trackers.py   online adapters around jointglmb_gms_python_fast / jointlmb_gms_python_fast + GNN baseline
  shims.py      pure-Python `murty` / `gibbs`, used only if the compiled packages are missing
  metrics.py    GOSPA, OSPA, OSPA(2), CLEAR-MOT, IDF1, track recall/precision, detector stats
  runner.py     run_experiment(), sweep()
  viz.py, ui.py plots, frame viewer, animation, ipywidgets control panel
```

Requirements: numpy, scipy, matplotlib, pandas, opencv-python (imported by the repo code), ipywidgets.

## Track-before-detect comparison (`tbd_vs_rfs_comparison.ipynb`)

IE-PHPMHT (arXiv:2504.20526), from the `track_before_detect_exp` project's `iephpmht` package, against the
detect-then-track pipelines on the same drone scenes (default SNR 2, clutter 1–5 px). Every algorithm is calibrated
on training seeds, then compared on held-out seeds for metrics and end-to-end latency.

- `dronebench/tbd.py`: adapter (frame source, coordinate convention, empirical-Bayes priors, `gamma_mix`, `p_s` override).
  It finds the package through `$IEPHPMHT_ROOT` or `../../track_before_detect_exp`.
- `dronebench/calibrate.py`: random + local search (`search`), `Bench` (cached detections, pre-roll),
  `calibrate_detection`, `calibrate_tbd`, `compare`, JSON persistence.
- `run_calibration.py <GLMB-gibbs|LMB-gibbs|GNN|IE-PHPMHT> [--refine --local N]`: writes `calib/<algo>.json`.
  The four can run in parallel, in about 10–30 min each on this machine.
- `make_tbd_notebook.py`: regenerates the notebook source.
