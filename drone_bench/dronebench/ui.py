"""ipywidgets control panel: change scene / detector / tracker knobs and re-run."""
from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt

from . import viz
from .runner import run_experiment


def playground(scenario, detector, trackers, state=None):
    """Interactive panel. The last experiment is stored in `state['ex']` (pass a dict to keep it)."""
    import ipywidgets as w
    from IPython.display import display

    state = {} if state is None else state
    sty = dict(style={"description_width": "150px"}, layout=w.Layout(width="420px"))
    snr = w.FloatRangeSlider(value=(scenario.snr, scenario.snr) if isinstance(scenario.snr, (int, float))
                             else tuple(scenario.snr), min=1, max=5, step=0.25, description="target SNR", **sty)
    size = w.FloatRangeSlider(value=tuple(scenario.target_size_px), min=1, max=5, step=0.5,
                              description="target FWHM px", **sty)
    ntg = w.IntRangeSlider(value=(scenario.n_targets_init, scenario.n_targets_init)
                           if isinstance(scenario.n_targets_init, int) else tuple(scenario.n_targets_init),
                           min=0, max=12, description="initial drones", **sty)
    birth = w.FloatSlider(value=scenario.birth_rate, min=0, max=0.2, step=0.01, description="drone births/frame", **sty)
    nfr = w.IntSlider(value=scenario.n_frames, min=20, max=200, step=10, description="frames", **sty)
    ncl = w.IntSlider(value=scenario.n_static_clutter, min=0, max=200, step=5, description="persistent clutter", **sty)
    mov = w.FloatSlider(value=scenario.moving_clutter_frac, min=0, max=1, step=0.05, description="moving clutter frac", **sty)
    tcl = w.FloatSlider(value=scenario.transient_clutter_rate, min=0, max=60, step=1, description="transient clutter/frame", **sty)
    csz = w.FloatRangeSlider(value=tuple(scenario.clutter_size_px), min=1, max=25, step=1, description="clutter FWHM px", **sty)
    csnr = w.FloatRangeSlider(value=tuple(scenario.clutter_snr), min=0.2, max=8, step=0.1, description="clutter SNR", **sty)
    thr = w.FloatSlider(value=detector.threshold, min=2, max=8, step=0.25, description="detector threshold σ", **sty)
    bgs = w.Dropdown(options=["running", "none"], value=detector.bg_subtract, description="background subtract", **sty)
    sg = w.FloatSlider(value=detector.size_gate_sigma or 0, min=0, max=8, step=0.25,
                       description="size gate σ (0=off)", **sty)
    algos = w.SelectMultiple(options=list(trackers), value=tuple(trackers), description="trackers", rows=len(trackers), **sty)
    pd_ = w.FloatSlider(value=next(iter(trackers.values())).P_D, min=0.3, max=0.99, step=0.01, description="tracker P_D", **sty)
    sv = w.FloatSlider(value=next(iter(trackers.values())).sigma_v, min=0.05, max=3, step=0.05, description="tracker σ_v px/f²", **sty)
    hup = w.IntSlider(value=next(iter(trackers.values())).H_upd, min=10, max=1000, step=10, description="H_upd (hyp/samples)", **sty)
    seed = w.IntText(value=0, description="seed", **sty)
    btn = w.Button(description="Run", button_style="primary", icon="play")
    out = w.Output()

    def run(_):
        with out:
            out.clear_output()
            scn = replace(scenario, snr=tuple(snr.value), target_size_px=tuple(size.value),
                          n_targets_init=tuple(ntg.value), birth_rate=birth.value, n_frames=nfr.value,
                          n_static_clutter=ncl.value, moving_clutter_frac=mov.value,
                          transient_clutter_rate=tcl.value, clutter_size_px=tuple(csz.value),
                          clutter_snr=tuple(csnr.value))
            det = replace(detector, threshold=thr.value, bg_subtract=bgs.value,
                          size_gate_sigma=(sg.value or None))
            trk = {n: replace(trackers[n], P_D=pd_.value, sigma_v=sv.value, H_upd=hup.value) for n in algos.value}
            ex = run_experiment(scn, det, trk, seed=seed.value)
            state["ex"] = ex
            display(ex.table()[["GOSPA", "OSPA2", "IDF1", "MOTA", "track_recall", "track_precision",
                                "ID_switches", "lat_mean_ms", "lat_p95_ms"]].round(3))
            viz.plot_summary(ex.table())
            viz.plot_trajectories(ex)
            viz.plot_metrics(ex)
            viz.plot_latency(ex)
            plt.show()
            viz.frame_viewer(ex)

    btn.on_click(run)
    col1 = w.VBox([w.HTML("<b>Drones</b>"), snr, size, ntg, birth, nfr])
    col2 = w.VBox([w.HTML("<b>Clutter</b>"), ncl, mov, tcl, csz, csnr])
    col3 = w.VBox([w.HTML("<b>Detector</b>"), thr, bgs, sg, w.HTML("<b>Trackers</b>"), algos, pd_, sv, hup, seed, btn])
    display(w.VBox([w.HBox([col1, col2]), col3, out]))
    return state
