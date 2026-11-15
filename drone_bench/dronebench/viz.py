"""Plots, frame viewer and animations."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

ALGO_COLORS = plt.get_cmap("tab10").colors
ID_CMAP = plt.get_cmap("tab20")


def _algo_color(i):
    return ALGO_COLORS[i % len(ALGO_COLORS)]


def _id_color(i):
    return ID_CMAP((int(i) * 7) % 20)


def _vlim(img):
    lo, hi = np.percentile(img, [1, 99.7])
    return lo, hi


# ----------------------------------------------------------------- single frame
def draw_frame(ax, ex, k, algo=None, show_dets=True, show_truth=True, trail=15, zoom=None):
    seq = ex.seq
    img = seq.frames[k]
    lo, hi = _vlim(img)
    ax.clear()
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    t = seq.truth[k]
    if show_dets and ex.Z[k].shape[1]:
        ax.plot(ex.Z[k][0], ex.Z[k][1], "x", color="tab:red", ms=4, mew=1, label="detections")
    if show_truth:
        for i, (x, y) in zip(t["ids"], t["pos"]):
            ax.add_patch(plt.Circle((x, y), 6, fill=False, ec="lime", lw=1.2))
        if len(t["ids"]):
            ax.plot([], [], "o", mfc="none", mec="lime", label="truth")
    if algo is not None:
        run = ex.runs[algo]
        hist = {}
        for j in range(max(0, k - trail), k + 1):
            for i, x in zip(run.ids[j], run.pos[j]):
                hist.setdefault(int(i), []).append(x)
        for i, x in zip(run.ids[k], run.pos[k]):
            col = _id_color(i)
            h = np.array(hist.get(int(i), [x]))
            ax.plot(h[:, 0], h[:, 1], "-", color=col, lw=1.2, alpha=0.9)
            ax.add_patch(plt.Rectangle((x[0] - 5, x[1] - 5), 10, 10, fill=False, ec=col, lw=1.5))
            ax.text(x[0] + 6, x[1] - 6, str(i), color=col, fontsize=8, weight="bold")
        ax.plot([], [], "s", mfc="none", mec="w", label=f"{algo} tracks")
    ax.set_xlim(-0.5, seq.cfg.width - 0.5)
    ax.set_ylim(seq.cfg.height - 0.5, -0.5)
    if zoom is not None:
        cx, cy, half = zoom
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy + half, cy - half)
    nt, ne = len(t["ids"]), (len(ex.runs[algo].ids[k]) if algo else 0)
    lat = f" | {ex.runs[algo].latency[k] * 1e3:.0f} ms" if algo else ""
    ax.set_title(f"frame {k}: {nt} drones, {ex.Z[k].shape[1]} detections"
                 + (f", {ne} tracks" if algo else "") + lat, fontsize=9)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])


def show_frames(ex, ks, algo=None, cols=3, size=4.2):
    rows = int(np.ceil(len(ks) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(size * cols, size * rows * ex.seq.cfg.height / ex.seq.cfg.width))
    for ax, k in zip(np.ravel(axs), ks):
        draw_frame(ax, ex, k, algo)
    for ax in np.ravel(axs)[len(ks):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def frame_viewer(ex):
    """Interactive viewer: play / scrub frames, pick algorithm, toggle overlays."""
    import ipywidgets as w
    from IPython.display import display

    algos = ["(none)"] + list(ex.runs)
    k = w.IntSlider(0, 0, ex.seq.K - 1, description="frame", continuous_update=False,
                    layout=w.Layout(width="420px"))
    play = w.Play(min=0, max=ex.seq.K - 1, interval=250)
    w.jslink((play, "value"), (k, "value"))
    algo = w.Dropdown(options=algos, value=algos[1] if len(algos) > 1 else algos[0], description="tracker")
    dets = w.Checkbox(True, description="detections")
    truth = w.Checkbox(True, description="truth")
    trail = w.IntSlider(15, 0, 60, description="trail")
    out = w.Output()

    def redraw(*_):
        with out:
            out.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(9, 9 * ex.seq.cfg.height / ex.seq.cfg.width))
            draw_frame(ax, ex, k.value, None if algo.value == "(none)" else algo.value,
                       dets.value, truth.value, trail.value)
            plt.show()

    for c in (k, algo, dets, truth, trail):
        c.observe(redraw, "value")
    redraw()
    display(w.VBox([w.HBox([play, k]), w.HBox([algo, dets, truth, trail]), out]))


def animate(ex, algo=None, fps=6, frames=None, size=7):
    """Returns a matplotlib FuncAnimation (use .to_jshtml() or save_animation)."""
    ks = list(range(ex.seq.K)) if frames is None else list(frames)
    fig, ax = plt.subplots(figsize=(size, size * ex.seq.cfg.height / ex.seq.cfg.width))
    anim = animation.FuncAnimation(fig, lambda i: draw_frame(ax, ex, ks[i], algo), frames=len(ks),
                                   interval=1000 / fps)
    plt.close(fig)
    return anim


def save_animation(anim, path, fps=6):
    if str(path).endswith(".gif"):
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(path, fps=fps)
    return path


# --------------------------------------------------------------- trajectories
def plot_trajectories(ex, algos=None):
    algos = list(ex.runs) if algos is None else algos
    seq = ex.seq
    n = len(algos)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.2 * rows * seq.cfg.height / seq.cfg.width + 0.6),
                            squeeze=False)
    gt = {}
    for t in seq.truth:
        for i, x in zip(t["ids"], t["pos"]):
            gt.setdefault(int(i), []).append(x)
    for ax, name in zip(axs.ravel(), algos):
        ax.imshow(seq.frames[0], cmap="gray", alpha=0.35, vmin=_vlim(seq.frames[0])[0], vmax=_vlim(seq.frames[0])[1])
        for p in gt.values():
            p = np.array(p)
            ax.plot(p[:, 0], p[:, 1], "--", color="k", lw=2.2, alpha=0.8)
        tr = {}
        for ids, pos in zip(ex.runs[name].ids, ex.runs[name].pos):
            for i, x in zip(ids, pos):
                tr.setdefault(int(i), []).append(x)
        for i, p in tr.items():
            p = np.array(p)
            ax.plot(p[:, 0], p[:, 1], "-", color=_id_color(i), lw=1.4)
        s = ex.runs[name].summary
        ax.set_title(f"{name}: {len(tr)} tracks vs {len(gt)} drones\n"
                     f"GOSPA {s['GOSPA']:.1f}  IDF1 {s['IDF1']:.2f}", fontsize=9)
        ax.set_xlim(0, seq.cfg.width)
        ax.set_ylim(seq.cfg.height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Truth (black dashed) vs estimated tracks (coloured by label)", fontsize=10)
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------- metrics
def plot_metrics(ex):
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    K = np.arange(ex.seq.K)
    for j, (name, r) in enumerate(ex.runs.items()):
        c = _algo_color(j)
        fm = r.frame_metrics
        axs[0, 0].plot(K, fm["gospa"], color=c, label=name)
        axs[0, 1].plot(K, fm["ospa2"], color=c, label=name)
        axs[1, 0].step(K, fm["card_est"], where="mid", color=c, label=name)
    axs[1, 0].step(K, next(iter(ex.runs.values())).frame_metrics["card_true"], where="mid",
                   color="k", lw=2.5, label="truth")
    names = list(ex.runs)
    comp = np.array([[ex.runs[n].summary[k] ** 2 for k in ("GOSPA_loc", "GOSPA_missed", "GOSPA_false")]
                     for n in names])
    bottom = np.zeros(len(names))
    for i, lab in enumerate(["localisation", "missed", "false"]):
        axs[1, 1].bar(names, comp[:, i], bottom=bottom, label=lab)
        bottom += comp[:, i]
    axs[0, 0].set_title("GOSPA per frame (c=10, p=2, α=2)")
    axs[0, 1].set_title("OSPA(2) track metric (sliding window)")
    axs[1, 0].set_title("Cardinality: estimated vs true")
    axs[1, 1].set_title("Mean squared GOSPA split by error type")
    axs[1, 1].tick_params(axis="x", rotation=20)
    for ax in axs.ravel():
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axs[1, 0].set_xlabel("frame")
    axs[0, 0].set_xlabel("frame")
    axs[0, 1].set_xlabel("frame")
    fig.tight_layout()
    return fig


def plot_summary(table, metrics=("GOSPA", "OSPA2", "IDF1", "MOTA", "track_recall", "track_precision")):
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.0 * len(metrics), 3.2))
    for ax, m in zip(axs, metrics):
        ax.bar(table.index, table[m], color=[_algo_color(i) for i in range(len(table))])
        ax.set_title(m + ("  (lower=better)" if m in ("GOSPA", "OSPA2", "OSPA", "card_err") else ""), fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_detector(ex):
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.2))
    K = np.arange(ex.seq.K)
    axs[0].plot(K, ex.det_pd, "o-", ms=3)
    axs[0].set_ylim(0, 1.05)
    axs[0].set_title(f"Detector Pd per frame (mean {np.nanmean(ex.det_pd):.2f})")
    axs[1].plot(K, ex.det_fa, "o-", ms=3, color="tab:red")
    axs[1].set_title(f"False alarms per frame (mean {ex.det_fa.mean():.1f})")
    axs[2].plot(K, ex.det_latency * 1e3, color="tab:gray")
    axs[2].set_title(f"Detector latency (mean {ex.det_latency.mean() * 1e3:.1f} ms)")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.set_xlabel("frame")
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------- latency
def plot_latency(ex, budget_ms=33.3):
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    nmeas = np.array([z.shape[1] for z in ex.Z])
    names = list(ex.runs)
    data = []
    for j, (name, r) in enumerate(ex.runs.items()):
        c = _algo_color(j)
        lat = r.latency * 1e3
        data.append(lat[np.isfinite(lat)])
        axs[0, 0].plot(lat, color=c, label=name)
        axs[1, 0].scatter(nmeas, lat, s=10, color=c, alpha=0.7, label=name)
        axs[1, 1].plot(r.n_hyp, color=c, label=name)
    axs[0, 0].axhline(budget_ms, color="k", ls=":", label=f"{budget_ms:.0f} ms budget")
    axs[0, 1].boxplot(data, labels=names, showfliers=True, whis=(5, 95))
    axs[0, 1].axhline(budget_ms, color="k", ls=":")
    axs[0, 0].set_title("Tracker latency per frame")
    axs[0, 1].set_title("Latency distribution (whiskers p5-p95)")
    axs[1, 0].set_title("Latency vs number of measurements")
    axs[1, 1].set_title("Filter size (GLMB hypotheses / LMB & GNN tracks)")
    axs[0, 0].set_xlabel("frame")
    axs[1, 1].set_xlabel("frame")
    axs[1, 0].set_xlabel("detections in frame")
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.set_ylabel("ms")
        ax.set_yscale("log")
    axs[0, 1].tick_params(axis="x", rotation=20)
    for ax in axs.ravel():
        ax.grid(alpha=0.3, which="both")
    for ax in (axs[0, 0], axs[1, 0], axs[1, 1]):
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- sweeps
def plot_sweep(df, param, metrics=("GOSPA", "IDF1", "track_recall", "MOTA", "lat_mean_ms", "det_fa")):
    fig, axs = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.3))
    for ax, m in zip(axs, metrics):
        for j, (name, g) in enumerate(df.groupby("algorithm", sort=False)):
            s = g.groupby(param)[m].agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"].fillna(0), marker="o", capsize=3,
                        color=_algo_color(j), label=name)
        ax.set_title(m, fontsize=9)
        ax.set_xlabel(param.split(".")[-1])
        ax.grid(alpha=0.3)
        if m.startswith("lat"):
            ax.set_yscale("log")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    return fig
