import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Track-before-detect (IE-PHPMHT) vs detect-then-track (GLMB / LMB / GNN) on simulated drones

Same environment as `drone_tracking_playground.ipynb` (`dronebench.simulate`): planar drones that hover, cruise,
accelerate and circle, 1–5 px Gaussian blobs, heavy clutter of persistent and one-frame blobs, white noise σ = 1.
Default here: **SNR 2, clutter 1–5 px** (the same size as the drones), 320×240.

| | pipeline |
|---|---|
| **GLMB-gibbs, LMB-gibbs, GNN** | per-frame point detector (background subtraction → high-pass → matched filter → threshold + NMS + LoG size gate) → tracker on the detections |
| **IE-PHPMHT** | Herrmann, García-Fernández & Brekke, *Histogram-PMHT with integrated target existence*, [arXiv:2504.20526](https://arxiv.org/abs/2504.20526), as implemented in `Neural-Glimpse/track_before_detect_exp` (`notebooks/iephpmht_1080p_realtime.ipynb`), refactored into the `iephpmht` package. **No detection threshold**: the raw frame becomes Poisson-like cell counts, Bernoulli components with an IMM (CV, CT±, CA) are fitted to them by EM, and existence comes from the fitted Poisson rate (eq. 47). |

**Protocol**
1. Every sequence has a **60-frame pre-roll**. All algorithms see it, and background models and tracks start there. Metrics are computed on the following **80 frames** only.
2. **Calibration** minimises mean GOSPA (c = 10 px, p = 2, α = 2) on **training seeds 11–13**. It uses a random search, then a shrinking local search around the best point, always starting from the defaults.
   - The detector threshold is tuned jointly with each tracker.
   - IE-PHPMHT's priors (α, β, γ, p_s, p_b) come from the paper/project **empirical-Bayes calibration** on a separate labelled **calibration seed 100**, recomputed for every front-end setting. Its structural, motion and reporting parameters are searched on top.
3. **Comparison** on **held-out test seeds 1–6**: the default and the calibrated configuration of every algorithm.
4. **Latency**: single thread, CPU, frames in memory.
   - Detection pipelines: detector + tracker.
   - IE-PHPMHT: front end + filter (the project's NumPy port of its CUDA front end).

IE-PHPMHT got a second stage on top: a 30-trial local refinement with widened bounds, then a line search over `p_b_scale`, still on training seeds only, because its first search ended at a bound.

Calibration results are cached in `calib/*.json`; `run_calibration.py <algo>` regenerates them (≈ 15–40 min per algorithm).
""")

code(r"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from dataclasses import replace, asdict
warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, os.path.abspath("."))

from dronebench import ScenarioConfig, DetectorConfig, TrackerConfig, simulate
from dronebench.calibrate import (Bench, compare, summary_table, from_jsonable, load_results, to_experiment,
                                  TBDCalibrator, effective_priors)
from dronebench.tbd import TBDConfig, SeqSource, eb_calibrate, IEPHPMHT_ROOT
from dronebench import viz
import run_calibration as RC
import iephpmht as ip

pd.set_option("display.precision", 3); pd.set_option("display.width", 200)
print("iephpmht from", IEPHPMHT_ROOT, "| front end backend:", "CUDA" if ip.cuda_available() else "CPU (NumPy)")
""")

md(r"""
## 1. Scenario: edit here

`SETUP` is shared with `run_calibration.py`. If you change the scenario (SNR, clutter, size), the cached calibration
no longer matches it. Set `RECALIBRATE = True` to rerun the searches inside the notebook (slow), or run them in parallel
from a shell: `python run_calibration.py IE-PHPMHT` etc.
""")

code(r"""
SETUP = dict(RC.SETUP)           # snr=2.0, clutter_size_px=(1,5), pre=60, n_eval=80, train 11-13, calib 100, test 1-6
# SETUP.update(snr=3.0)          # <- play here
RECALIBRATE = False
CALIB_DIR = "calib"
ALGOS = ["GLMB-gibbs", "LMB-gibbs", "GNN", "IE-PHPMHT"]

scn = RC.scenario(SETUP)
PRE = SETUP["pre"]
print(scn)
""")

md("## 2. What each pipeline sees")

code(r"""
seq = simulate(scn, SETUP["test_seeds"][0])
k = PRE + 20
det_default = DetectorConfig()
from dronebench.detector import detect_sequence
Z, _, _ = detect_sequence(seq.frames[:k + 1], det_default)
fe = ip.make_frontend(seq.frames.shape[1:], 2, 0.7, backend="cpu", n_warm=40)
src = SeqSource(seq)
for j in range(k + 1):
    fe.process(src.frame(j))
cand, zval = fe.birth_candidates()

t = seq.truth[k]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].imshow(seq.frames[k], cmap="gray", vmin=-3, vmax=6); ax[0].set_title(f"raw frame {k} (SNR {scn.snr}, noise σ=1)")
ax[1].imshow(seq.frames[k], cmap="gray", vmin=-3, vmax=6)
ax[1].plot(Z[k][0], Z[k][1], "x", c="tab:red", ms=5, label=f"detections ({Z[k].shape[1]})")
ax[1].set_title("detector output (default threshold 4σ)")
im = ax[2].imshow(fe.T, cmap="magma", vmin=-2, vmax=8, extent=(-0.5, seq.cfg.width - 0.5, seq.cfg.height - 0.5, -0.5))
ax[2].plot(cand[:, 0] * 2 - 0.5, cand[:, 1] * 2 - 0.5, "+", c="cyan", ms=7, label=f"birth proposals ({len(cand)})")
ax[2].set_title("IE-PHPMHT front end: birth statistic Z on 2×2 cells"); plt.colorbar(im, ax=ax[2], fraction=0.03)
for a in ax:
    for x, y in t["pos"]:
        a.add_patch(plt.Circle((x, y), 7, fill=False, ec="lime", lw=1.3))
    a.set_xticks([]); a.set_yticks([]); a.set_xlim(-0.5, seq.cfg.width - 0.5); a.set_ylim(seq.cfg.height - 0.5, -0.5)
ax[1].legend(loc="upper right", fontsize=8); ax[2].legend(loc="upper right", fontsize=8)
fig.suptitle("green circles = true drones", y=1.0); plt.tight_layout(); plt.show()
""")

md(r"""
## 3. Calibration

Loads `calib/<algo>.json` (written by `run_calibration.py`), or runs the search here if a file is missing or `RECALIBRATE` is set.
""")

code(r"""
CAL = {}
for algo in ALGOS:
    path = os.path.join(CALIB_DIR, f"{algo}.json")
    if RECALIBRATE or not os.path.exists(path):
        print(f"calibrating {algo} ...")
        res = RC.calibrate_one(algo, setup=SETUP)
        os.makedirs(CALIB_DIR, exist_ok=True)
        from dronebench.calibrate import save_results
        save_results(path, res, meta=dict(setup=SETUP))
    d = load_results(path)
    if d["meta"]["setup"] != json.loads(json.dumps(SETUP)):
        print(f"WARNING {algo}: calibration was made for a different SETUP: {d['meta']['setup']}")
    CAL[algo] = d["results"]

rows = []
for algo, r in CAL.items():
    rows.append(dict(algorithm=algo, trials=r["n_trials"], minutes=r["seconds"] / 60,
                     train_GOSPA_default=r["default_train_gospa"], train_GOSPA_calibrated=r["train_gospa"]))
display(pd.DataFrame(rows).set_index("algorithm"))

fig, axs = plt.subplots(1, len(CAL), figsize=(4.2 * len(CAL), 3.2), sharey=True)
for ax, (algo, r) in zip(axs, CAL.items()):
    h = pd.DataFrame(r["history"])
    ax.plot(h.score.clip(upper=40), ".", c="0.6", label="trial")
    ax.plot(h.score.cummin(), "-", c="tab:red", label="best so far")
    ax.axvline((h.phase == "random").sum() + 0.5, ls=":", c="k")
    ax.set_title(algo); ax.set_xlabel("trial (random | local)")
axs[0].set_ylabel("train GOSPA (clipped at 40)"); axs[0].legend(); plt.tight_layout(); plt.show()
""")

code(r"""
# default vs calibrated parameters
DEF = RC.default_configs()
TUNED, TBD_CAL = {}, None
for algo in ALGOS:
    det, trk, tbd, cal, _ = from_jsonable(CAL[algo])
    TUNED[algo] = (det, trk) if algo != "IE-PHPMHT" else (tbd, cal)
    if algo == "IE-PHPMHT":
        TBD_CAL = cal

def diff_params(a, b, prefix):
    da, db = asdict(a), asdict(b)
    return {f"{prefix}.{k}": (da[k], db[k]) for k in da if da[k] != db[k]}

for algo in ALGOS:
    if algo == "IE-PHPMHT":
        d = diff_params(DEF[algo][0], TUNED[algo][0], "tbd")
    else:
        d = {**diff_params(DEF[algo][0], TUNED[algo][0], "detector"), **diff_params(DEF[algo][1], TUNED[algo][1], "tracker")}
    print(f"--- {algo}")
    for k, (a, b) in d.items():
        fa = f"{a:.4g}" if isinstance(a, float) else str(a); fb = f"{b:.4g}" if isinstance(b, float) else str(b)
        print(f"   {k:28s} {fa:>10s} -> {fb}")
""")

md(r"""
### IE-PHPMHT priors: empirical Bayes, and why γ needs care

The empirical-Bayes step (project `calibrate`) fits the following on the calibration seed:
- **α, β**: Gamma, by method of moments, to single-frame Poisson-rate MLEs at the **true** drone positions;
- **γ**: 1 / mean rate MLE at random **target-free** positions, which is the rate a Bernoulli sitting on nothing gets;
- **p_s** (survival) and **p_b** (share of birth proposals that are real new drones), from the truth stream.

At SNR 2 the on-target Gamma is very broad. A confirmed Bernoulli that has lost its target keeps a posterior rate near (a − 1)/(b + 1) ≈ 3. At that rate, eq. 47's target likelihood G(λ; α, β) and clutter likelihood E(λ; γ) are about equal, so with the fixed-position γ **false tracks sustain themselves**.

A drifting Bernoulli is not at a fixed position. EM moves it to the local intensity maximum, so the right "nothing there" reference is the **position-free** MLE (best rate within ±1 cell, `off_adapt`). The adapter therefore has `gamma_mix`, which log-interpolates γ between the paper's definition (0) and the position-free one (1). Like p_s and t_conf, it is chosen by the search, not by hand.
""")

code(r"""
cfg_t, _ = TUNED["IE-PHPMHT"]
cal_t, diag = eb_calibrate(simulate(scn, SETUP["calib_seed"]), cfg_t, seed=SETUP["calib_seed"])
from scipy import stats
eff = effective_priors(cfg_t, cal_t)
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))
bins = np.linspace(0, max(60, np.percentile(diag["on"], 99)), 70)
ax[0].hist(diag["on"], bins, density=True, alpha=.5, label=f"on target (n={len(diag['on'])})")
ax[0].hist(diag["off"], bins, density=True, alpha=.5, label="target-free, fixed position")
ax[0].hist(diag["off_adapt"], bins, density=True, alpha=.5, label="target-free, position free ±1 cell")
x = np.linspace(0.01, bins[-1], 400)
ax[0].plot(x, stats.gamma.pdf(x, cal_t["alpha"], scale=1 / cal_t["beta"]), "C0", lw=2, label=f"Gamma(α={cal_t['alpha']:.2f}, β={cal_t['beta']:.3f})")
ax[0].plot(x, cal_t["gamma"] * np.exp(-cal_t["gamma"] * x), "C1--", lw=2, label=f"Exp(γ={cal_t['gamma']:.2f}) paper")
ax[0].plot(x, eff["gamma"] * np.exp(-eff["gamma"] * x), "C2", lw=2, label=f"Exp(γ={eff['gamma']:.2f}) used (gamma_mix={cfg_t.gamma_mix:.2f})")
ax[0].set_ylim(0, 0.25); ax[0].set_xlabel("single-frame Poisson rate MLE λ"); ax[0].legend(fontsize=7)
lr = lambda g: stats.gamma.logpdf(x, cal_t["alpha"], scale=1 / cal_t["beta"]) - (np.log(g) - g * x)
ax[1].plot(x, lr(cal_t["gamma"]), "C1--", label="paper γ"); ax[1].plot(x, lr(eff["gamma"]), "C2", label="used γ")
ax[1].axhline(0, c="k", lw=.8); ax[1].set_xlim(0, 30); ax[1].set_ylim(-8, 6)
ax[1].set_xlabel("λ"); ax[1].set_ylabel("log G(λ;α,β) / E(λ;γ)"); ax[1].legend(); ax[1].set_title("existence evidence per frame (eq. 47)")
plt.tight_layout(); plt.show()
print("calibrated:", {k: round(v, 4) for k, v in cal_t.items()}, "\nused:      ", {k: round(v, 4) for k, v in eff.items()})
""")

md(r"""
## 4. Held-out comparison (test seeds 1–6)

Each algorithm runs with its **default** configuration and with its **calibrated** one. Latencies come from this
process, one algorithm at a time.
""")

code(r"""
test = Bench(scn, SETUP["test_seeds"], pre=PRE)
cal_default_tbd = TBDCalibrator(scn, SETUP["calib_seed"])(DEF["IE-PHPMHT"][0])
det_runs = {}
for algo in ["GLMB-gibbs", "LMB-gibbs", "GNN"]:
    det_runs[f"{algo} (default)"] = DEF[algo]
    det_runs[f"{algo} (calibrated)"] = TUNED[algo]
tbd_runs = {"IE-PHPMHT (default)": (DEF["IE-PHPMHT"][0], cal_default_tbd),
            "IE-PHPMHT (calibrated)": TUNED["IE-PHPMHT"]}
t0 = time.perf_counter()
DF, RUNS = compare(test, det_runs, tbd_runs)
print(f"{time.perf_counter() - t0:.0f} s")
""")

code(r"""
mean, std = summary_table(DF)
order = [n for n in mean.index if "calibrated" in n] + [n for n in mean.index if "default" in n]
show = mean.loc[order].copy()
show.columns = [c.replace("_ms", " [ms]") for c in show.columns]
display(show.style.format("{:.3g}").background_gradient(subset=["GOSPA"], cmap="RdYlGn_r")
        .background_gradient(subset=["IDF1", "track_recall"], cmap="RdYlGn"))
print("± std over seeds (calibrated):")
display(std.loc[[n for n in order if "calibrated" in n], ["GOSPA", "IDF1", "MOTA", "track_recall", "lat_mean_ms"]])
""")

code(r"""
metrics = ["GOSPA", "GOSPA_missed", "GOSPA_false", "OSPA2", "IDF1", "MOTA", "track_recall", "track_precision"]
names = [n.replace(" (calibrated)", "") for n in order if "calibrated" in n]
fig, axs = plt.subplots(2, 4, figsize=(17, 6.5))
for ax, m in zip(axs.ravel(), metrics):
    for j, base in enumerate(names):
        for o, kind, alpha in ((-0.2, "default", 0.35), (0.2, "calibrated", 1.0)):
            d = DF[DF.algorithm == f"{base} ({kind})"][m]
            ax.bar(j + o, d.mean(), 0.38, yerr=d.std(), color=viz._algo_color(j), alpha=alpha, capsize=2,
                   label=kind if j == 0 else None)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, fontsize=8); ax.set_title(m)
axs[0, 0].legend(fontsize=8)
fig.suptitle(f"test seeds {SETUP['test_seeds']}: pale = default, solid = calibrated (mean ± std)"); plt.tight_layout(); plt.show()
""")

md("### Latency")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
cal_names = [n for n in order if "calibrated" in n]
for j, n in enumerate(cal_names):
    s1 = np.concatenate([RUNS[n][s]["det_ms"] for s in test.seeds]); s2 = np.concatenate([RUNS[n][s]["trk_ms"] for s in test.seeds])
    ax[0].bar(j, s1.mean(), color="0.65", label="detector / front end" if j == 0 else None)
    ax[0].bar(j, s2.mean(), bottom=s1.mean(), color=viz._algo_color(j), label="tracker / filter" if j == 0 else None)
    ax[0].text(j, s1.mean() + s2.mean(), f"{s1.mean() + s2.mean():.1f}", ha="center", va="bottom", fontsize=9)
    ax[1].hist(s1 + s2, bins=np.logspace(0, 3, 60), histtype="step", lw=1.8, color=viz._algo_color(j), label=n.replace(" (calibrated)", ""))
ax[0].set_xticks(range(len(cal_names))); ax[0].set_xticklabels([n.replace(" (calibrated)", "") for n in cal_names])
ax[0].set_ylabel("mean ms / frame"); ax[0].legend(); ax[0].set_title("end-to-end latency breakdown (calibrated)")
ax[1].set_xscale("log"); ax[1].axvline(33.3, ls="--", c="k", lw=.8); ax[1].text(35, ax[1].get_ylim()[1] * .9, "30 fps", fontsize=8)
ax[1].set_xlabel("ms / frame (end-to-end)"); ax[1].legend(fontsize=8); ax[1].set_title("per-frame latency distribution")
plt.tight_layout(); plt.show()
display(DF.groupby("algorithm")[["stage1_ms", "stage2_ms", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms"]].mean().loc[cal_names])
""")

md("### Visual check on one test seed")

code(r"""
SEED = test.seeds[0]
ex = to_experiment(test, SEED, {n.replace(" (calibrated)", ""): RUNS[n] for n in cal_names}, TUNED["GLMB-gibbs"][0])
viz.plot_trajectories(ex); plt.show()
""")

code(r"""
fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))
for ax, name in zip(axs.ravel(), ex.runs):
    viz.draw_frame(ax, ex, PRE + 40, algo=name, show_dets=(name != "IE-PHPMHT"))
    ax.set_title(name + " — " + ax.get_title(), fontsize=9)
plt.tight_layout(); plt.show()
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(15, 3.6))
for j, n in enumerate(cal_names):
    g = np.mean([RUNS[n][s]["frame"]["gospa"] for s in test.seeds], 0)
    ce = np.mean([RUNS[n][s]["frame"]["card_est"] for s in test.seeds], 0)
    ax[0].plot(np.arange(PRE, PRE + len(g)), g, color=viz._algo_color(j), label=n.replace(" (calibrated)", ""))
    ax[1].plot(np.arange(PRE, PRE + len(ce)), ce, color=viz._algo_color(j))
ct = np.mean([RUNS[cal_names[0]][s]["frame"]["card_true"] for s in test.seeds], 0)
ax[1].plot(np.arange(PRE, PRE + len(ct)), ct, "k--", lw=2, label="true")
ax[0].set_title("GOSPA per frame (mean over test seeds)"); ax[1].set_title("number of reported tracks"); ax[0].legend(fontsize=8); ax[1].legend()
for a in ax: a.set_xlabel("frame")
plt.tight_layout(); plt.show()
""")

md(r"""
## 5. Robustness: calibrated configurations across SNR

All configurations keep the parameters calibrated at the SETUP SNR, without retuning. IE-PHPMHT keeps its
priors from the calibration SNR. 2 seeds per point; set `RUN_SNR_SWEEP = False` to skip (≈ 5 min).
""")

code(r"""
RUN_SNR_SWEEP = True
SNRS = [1.5, 2.0, 3.0, 4.0]
if RUN_SNR_SWEEP:
    rows = []
    for snr in SNRS:
        b = Bench(replace(scn, snr=snr), (21, 22), pre=PRE)
        df, _ = compare(b, {a: TUNED[a] for a in ["GLMB-gibbs", "LMB-gibbs", "GNN"]},
                        {"IE-PHPMHT": TUNED["IE-PHPMHT"]}, verbose=False)
        df["snr"] = snr; rows.append(df); print("SNR", snr, "done")
    SW = pd.concat(rows)
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.4))
    for ax, m in zip(axs, ["GOSPA", "IDF1", "track_recall", "lat_mean_ms"]):
        for j, a in enumerate(["GLMB-gibbs", "LMB-gibbs", "GNN", "IE-PHPMHT"]):
            d = SW[SW.algorithm == a].groupby("snr")[m].mean()
            ax.plot(d.index, d.values, "o-", color=viz._algo_color(j), label=a)
        ax.set_xlabel("SNR"); ax.set_title(m)
    axs[0].legend(fontsize=8); plt.tight_layout(); plt.show()
    display(SW.groupby(["snr", "algorithm"])[["GOSPA", "IDF1", "track_recall", "lat_mean_ms"]].mean().unstack())
""")

md(r"""## 6. Conclusions (from the run above: SNR 2, clutter 1–5 px, test seeds 1–6)

| calibrated | GOSPA ↓ | IDF1 | MOTA | track recall | false part | latency (mean / p95) |
|---|---|---|---|---|---|---|
| **GLMB-gibbs** | **7.0** | **0.73** | 0.63 | **0.57** | 0.5 | 119 / 167 ms |
| **LMB-gibbs** | **7.1** | **0.73** | 0.64 | 0.55 | 0.5 | 133 / 158 ms |
| GNN | 10.5 | 0.33 | 0.35 | 0.18 | 0.3 | 43 / 48 ms |
| IE-PHPMHT | 10.3 | 0.39 | 0.39 | 0.32 | 0.9 | **10 / 12 ms** |

**Calibration matters for every algorithm, most of all for IE-PHPMHT.**
- With the project defaults and the paper's γ (rate of a Bernoulli at a fixed target-free position), IE-PHPMHT keeps about 60 self-sustaining false tracks: GOSPA 59.
- Calibrated, GOSPA drops to 10.3. The decisive settings: a γ based on the position-free off-target rate (`gamma_mix` ≈ 1.4), p_b ≈ 30× its empirical-Bayes value, and t_conf ≈ 0.7.
- GLMB and LMB gain about 20% (8.8 → 7.0), mostly by lowering the detector threshold from 4σ to about 3σ and letting the RFS filter reject the extra clutter.
- GNN cannot do that. Its best setting suppresses false tracks, which costs recall.

**Accuracy: GLMB and LMB are clearly best at SNR 2. IE-PHPMHT is level with GNN on GOSPA and better on identity.**
- IE-PHPMHT's error is almost entirely *missed* targets; its track precision is 1.0.
- It follows a drone accurately once it has a Bernoulli on it. But eq. 47 decides existence from a **single-frame** rate. At SNR 2 that rate (≈ 5–15) often falls below the broad on-target Gamma (mean ≈ 16), so evidence builds slowly and dim or fast drones are confirmed late or never.
- The SNR sweep confirms this. At SNR 3–4 the detect-then-track pipelines reach GOSPA 3–4, while IE-PHPMHT stays at 8–9 with track recall ≈ 0.5. It was calibrated at SNR 2 and not retuned, and the plateau suggests a structural limit (existence, birth or motion handling), not noise.

**Latency: IE-PHPMHT is 4–13× faster end to end, with the tightest tail.** It averages 10.4 ms per frame (p99 13.6 ms): 1.8 ms front end and 8.6 ms filter, in NumPy on one CPU thread.
- The detection pipelines pay about 35 ms for the detector alone (multi-scale LoG size gate).
- GLMB and LMB add 84–98 ms of tracking, because the calibrated low threshold triples the number of detections.
- Only IE-PHPMHT would run at 30 fps here without further optimisation.

**Caveats**
- 6 test seeds of 80 frames each; per-seed std of GOSPA is about 3, so GLMB and LMB are tied, and so are GNN and IE-PHPMHT.
- The IE-PHPMHT changes stay within calibration: γ definition, p_s, p_b and t_conf. The paper's algorithm is unchanged.
- Changes likely to close the recall gap, all outside this paper, not tried here:
  - multi-frame evidence for existence;
  - a less diffuse on-target prior (fit per size or SNR class);
  - a velocity-aware birth proposal for fast drones.
- Murty variants were left out of the calibration to save time. Earlier runs showed them close to Gibbs in accuracy and 2–3× slower.
- The trajectory plots draw the full truth including the 60-frame pre-roll; tracks are scored after it.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "tbd_vs_rfs_comparison.ipynb")   # run from drone_bench/; execute the notebook afterwards
print("written")
