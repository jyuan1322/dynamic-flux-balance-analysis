import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from lmfit import Parameters, minimize
from lmfit.models import LorentzianModel
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Standalone model helpers
# ---------------------------------------------------------------------------

def global_model_lmfit(params, x_data, n_peaks, n_traces):
    result = np.zeros((len(x_data), n_traces))
    for i in range(n_peaks):
        prefix = f"p{i}_"
        center = params[prefix + "center"].value
        sigma  = params[prefix + "sigma"].value
        model  = LorentzianModel(prefix=prefix)
        for t in range(n_traces):
            amp = params[f"{prefix}amp_t{t}"].value
            tp  = Parameters()
            tp.add(name=prefix + 'amplitude', value=amp)
            tp.add(name=prefix + 'center',    value=center)
            tp.add(name=prefix + 'sigma',     value=sigma)
            result[:, t] += model.eval(tp, x=x_data)
    return result


def global_model_lmfit_scaled(params, x_data, n_peaks, n_traces):
    result   = np.zeros((len(x_data), n_traces))
    peak_sum = np.zeros_like(x_data)
    for i in range(n_peaks):
        prefix = f"p{i}_"
        model  = LorentzianModel(prefix=prefix)
        tp = Parameters()
        tp.add(name=prefix + 'amplitude', value=1.0)
        tp.add(name=prefix + 'center',    value=params[prefix + "center"].value)
        tp.add(name=prefix + 'sigma',     value=params[prefix + "sigma"].value)
        peak_sum += model.eval(tp, x=x_data)
    baseline = params["baseline"].value
    for t in range(n_traces):
        result[:, t] = params[f"amp_t{t}"].value * peak_sum + baseline
    return result


def residual_global(params, x_data, y_data, n_peaks_total, trace_indices,
                    trace_weights=None):
    """
    Residual for global fit. Per-trace amplitudes, shared center/sigma.

    trace_weights : (n_active_traces,) array or None
        If provided, residuals for trace i are multiplied by trace_weights[i].
        Use weights = max_intensity^alpha so tall traces contribute more.
    """
    n_sub    = len(trace_indices)
    result   = np.zeros((len(x_data), n_sub))
    peak_sum = np.zeros_like(x_data)
    for i in range(n_peaks_total):
        prefix = f"p{i}_"
        model  = LorentzianModel(prefix=prefix)
        tp = Parameters()
        tp.add(name=prefix + 'amplitude', value=1.0)
        tp.add(name=prefix + 'center',    value=params[prefix + "center"].value)
        tp.add(name=prefix + 'sigma',     value=params[prefix + "sigma"].value)
        peak_sum += model.eval(tp, x=x_data)
    baseline = params["baseline"].value
    for i, t in enumerate(trace_indices):
        result[:, i] = params[f"amp_t{t}"].value * peak_sum + baseline
    residuals = y_data - result
    if trace_weights is not None:
        residuals *= trace_weights[np.newaxis, :]   # broadcast: (n_pts, n_traces)
    return residuals.ravel()


def _eval_lorentzians(fitted_params, x_full, n_peaks, active_indices,
                      n_traces_total, trace_mask):
    """
    Evaluate a fitted Lorentzian model on x_full for every trace.
    Active traces get their per-trace amplitude; masked traces get zero.
    """
    y_out = np.zeros((len(x_full), n_traces_total))
    if n_peaks == 0:
        return y_out

    peak_sum = np.zeros_like(x_full)
    for i in range(n_peaks):
        prefix = f"p{i}_"
        model  = LorentzianModel(prefix=prefix)
        tp = Parameters()
        tp.add(name=prefix + 'amplitude', value=1.0)
        tp.add(name=prefix + 'center',    value=fitted_params[prefix + "center"].value)
        tp.add(name=prefix + 'sigma',     value=fitted_params[prefix + "sigma"].value)
        peak_sum += model.eval(tp, x=x_full)

    bl = fitted_params["baseline"].value
    for t in range(n_traces_total):
        if not trace_mask[t]:
            continue
        y_out[:, t] = fitted_params[f"amp_t{t}"].value * peak_sum + bl

    return y_out


# ---------------------------------------------------------------------------
# Shared interactive fitting window
# ---------------------------------------------------------------------------

def _make_fitter_window(x_data, y_display, label,
                        active_indices, trace_mask, n_traces_total,
                        baseline, prominence_factor,
                        init_bounds, seed, real_times,
                        n_peaks_label, roi_color,
                        extra_buttons_factory=None):
    """
    Build and return a fully wired interactive fitting figure.

    Parameters
    ----------
    y_display : (N, T)
        Data to display AND fit (may already be background-subtracted).
    n_peaks_label : str
        Label for the peak-count textbox ("n_peaks" or "n_bkg_peaks").
    roi_color : str
        Colour for the ROI bound vlines ("g" for signal, "orange" for bkg).
    extra_buttons_factory : callable or None
        Called with (fig, ax) after the standard layout is done; can add
        extra axes/buttons (e.g. Skip button for the background window).
    baseline : float or None
        Fixed baseline; free if None.

    Returns
    -------
    fig, state dict, widget refs dict
    """
    n_traces = len(active_indices)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.38, left=0.1, right=0.85)
    ax.set_title(label)
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()

    # ---- display traces ----------------------------------------------------
    n_display = min(10, n_traces_total)
    if n_traces <= n_display:
        disp_idx = list(active_indices)
        inactive = np.where(~trace_mask)[0]
        rem = n_display - n_traces
        if rem > 0 and len(inactive) > 0:
            sel = np.linspace(0, len(inactive)-1, min(rem, len(inactive)), dtype=int)
            disp_idx.extend(inactive[sel].tolist())
        disp_idx = np.array(disp_idx)
    else:
        disp_idx = active_indices[np.linspace(0, n_traces-1, n_display, dtype=int)]

    colors     = plt.cm.viridis(np.linspace(0, 1, len(disp_idx)))
    data_lines = []
    fit_lines  = []
    for i, t in enumerate(disp_idx):
        is_active = trace_mask[t]
        ld, = ax.plot(x_data, y_display[:, t],
                      '.' if is_active else 'x',
                      alpha=0.6 if is_active else 0.2,
                      color=colors[i], markersize=3,
                      label=f"Trace {t}" + ("" if is_active else " (masked)"))
        lf, = ax.plot([], [], '-', color=colors[i], lw=2,
                      alpha=0.6 if is_active else 0.2)
        data_lines.append(ld)
        fit_lines.append(lf)
    ax.legend(fontsize=8, ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))

    # ---- sliders -----------------------------------------------------------
    axcolor = "lightgoldenrodyellow"
    ax_sl   = plt.axes([0.15, 0.25, 0.65, 0.02], facecolor=axcolor)
    ax_sr   = plt.axes([0.15, 0.21, 0.65, 0.02], facecolor=axcolor)

    if init_bounds is None:
        init_l, init_r = float(np.min(x_data)), float(np.max(x_data))
    else:
        init_l, init_r = init_bounds

    s_left  = Slider(ax_sl, "Lower PPM", np.min(x_data), np.max(x_data), valinit=init_l)
    s_right = Slider(ax_sr, "Upper PPM", np.min(x_data), np.max(x_data), valinit=init_r)

    vl = ax.axvline(s_left.val,  color=roi_color, linestyle="--", lw=2)
    vr = ax.axvline(s_right.val, color=roi_color, linestyle="--", lw=2)

    def _update_roi(val=None):
        l, r = sorted([s_left.val, s_right.val])
        vl.set_xdata([l, l])
        vr.set_xdata([r, r])
        fig.canvas.draw_idle()
    s_left.on_changed(_update_roi)
    s_right.on_changed(_update_roi)

    # ---- textboxes ---------------------------------------------------------
    ax_tb_n      = plt.axes([0.15, 0.11, 0.12, 0.04])
    ax_tb_nfev   = plt.axes([0.15, 0.06, 0.12, 0.04])
    ax_tb_wexp   = plt.axes([0.15, 0.01, 0.12, 0.04])
    tb_n    = TextBox(ax_tb_n,    n_peaks_label, initial="")
    tb_nfev = TextBox(ax_tb_nfev, "max_iters",   initial="")
    tb_wexp = TextBox(ax_tb_wexp, "weight_exp",  initial="0")

    # ---- buttons -----------------------------------------------------------
    ax_btn_fresh  = plt.axes([0.35, 0.11, 0.22, 0.04])
    ax_btn_pinned = plt.axes([0.35, 0.06, 0.22, 0.04])
    btn_fresh  = Button(ax_btn_fresh,  "Refit All FIDs")
    btn_pinned = Button(ax_btn_pinned, "Refit w/ Current Centroids")

    if extra_buttons_factory is not None:
        extra_buttons_factory(fig, ax)

    # ---- draggable center lines --------------------------------------------
    center_lines     = []
    center_overrides = []
    drag_state = {"active_idx": None, "last_x": None}

    def _make_center_lines(centers, preserve_overrides=False):
        old_ov = list(center_overrides) if preserve_overrides else []
        for ln in center_lines:
            ln.remove()
        center_lines.clear()
        center_overrides.clear()
        for i, c in enumerate(centers):
            ln = ax.axvline(c, color=roi_color, linestyle=":", lw=2,
                            alpha=0.85, picker=6, label=f"Peak {i+1} center")
            center_lines.append(ln)
            center_overrides.append(
                old_ov[i] if (preserve_overrides and i < len(old_ov)) else None
            )
        fig.canvas.draw_idle()

    def _get_centers():
        return [
            center_overrides[i] if center_overrides[i] is not None
            else center_lines[i].get_xdata()[0]
            for i in range(len(center_lines))
        ]

    def _on_press(event):
        if event.inaxes != ax:
            return
        for i, ln in enumerate(center_lines):
            hit, _ = ln.contains(event)
            if hit:
                drag_state["active_idx"] = i
                drag_state["last_x"]     = event.xdata
                ln.set_linewidth(3)
                fig.canvas.draw_idle()
                break

    def _on_motion(event):
        i = drag_state["active_idx"]
        if i is None or event.xdata is None:
            return
        new_x = center_lines[i].get_xdata()[0] + (event.xdata - drag_state["last_x"])
        center_lines[i].set_xdata([new_x, new_x])
        center_overrides[i] = new_x
        drag_state["last_x"] = event.xdata
        fig.canvas.draw_idle()

    def _on_release(event):
        i = drag_state["active_idx"]
        if i is not None:
            center_lines[i].set_linewidth(2)
            drag_state["active_idx"] = None
            drag_state["last_x"]     = None
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event",   _on_press)
    fig.canvas.mpl_connect("motion_notify_event",  _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)

    # ---- core fit ----------------------------------------------------------
    n_traces = len(active_indices)  # used inside _run_fit closure
    print(f"[fitter window] label={label!r:.60}  baseline={baseline}  n_traces={n_traces}")
    window_state = {"seed": seed, "lower_ppm_bound": init_l, "upper_ppm_bound": init_r}

    def _run_fit(center_init_mode="auto"):
        window_state["seed"] += 1
        np.random.seed(window_state["seed"])
        mode_label = "FRESH" if center_init_mode == "auto" else "PINNED"
        print(f"\n{'='*80}")
        print(f"FIT [{mode_label}] — seed {window_state['seed']}")
        print(f"{'='*80}\n")

        left, right = sorted([s_left.val, s_right.val])
        outer_left  = left  - 0.05
        outer_right = right + 0.05
        window_state["lower_ppm_bound"] = left
        window_state["upper_ppm_bound"] = right

        # mask = (x_data >= outer_left) & (x_data <= outer_right)
        mask = (x_data >= left) & (x_data <= right)
        if np.sum(mask) < 5:
            print("Too few points in region")
            return

        x_sub        = x_data[mask]
        y_sub_active = y_display[mask, :][:, active_indices]

        # ---- peak count / pinned centers -----------------------------------
        if center_init_mode == "pinned" and len(center_lines) > 0:
            pinned_centers = _get_centers()
            n_peaks = len(pinned_centers)
            tb_n.set_val(str(n_peaks))
        else:
            pinned_centers = []
            y_mean = np.mean(y_sub_active, axis=1)
            peaks_idx, _ = find_peaks(
                y_mean,
                prominence=prominence_factor * (y_mean.max() - y_mean.min())
            )
            if len(peaks_idx) == 0:
                peaks_idx = [np.argmax(y_mean)]
            if len(peaks_idx) > 10:
                dist = np.abs(x_sub[peaks_idx] - x_sub.mean())
                peaks_idx = [peaks_idx[k] for k in np.argsort(dist)[:10]]

            try:
                n_peaks = max(1, int(tb_n.text))
            except ValueError:
                n_peaks = len(peaks_idx)
                tb_n.set_val(str(n_peaks))

        print(f"Fitting {n_peaks} peak(s) across {n_traces} FIDs")
        print(f"ROI: [{left:.4f}, {right:.4f}] ppm")

        # ---- Parameters ----------------------------------------------------
        params = Parameters()
        for i in range(n_peaks):
            prefix = f"p{i}_"
            if center_init_mode == "pinned" and i < len(pinned_centers):
                c = float(np.clip(pinned_centers[i], x_sub.min(), x_sub.max()))
                params.add(name=prefix + "center", value=c, vary=False)
            else:
                if i < len(peaks_idx):
                    c = x_sub[peaks_idx[i]] + np.random.uniform(-0.05, 0.05)
                else:
                    frac = i / max(n_peaks - 1, 1)
                    c    = outer_left + frac * (outer_right - outer_left)
                    c   += np.random.uniform(-0.01, 0.01)
                params.add(name=prefix + "center",
                           value=np.clip(c, x_sub.min(), x_sub.max()),
                           min=x_sub.min(), max=x_sub.max())
            params.add(name=prefix + "sigma", value=0.005 * np.random.uniform(0.8, 1.2),
                       min=0.001, max=0.2)

        if baseline is not None:
            params.add(name="baseline", value=float(baseline), vary=False)
            print(f"Baseline fixed at pre-estimated value: {baseline:.5e}")
        else:
            # No pre-estimated baseline — fit it freely (legacy fallback)
            print("WARNING: no baseline provided — fitting baseline freely.")
            params.add(name="baseline", value=float(np.min(y_sub_active)), vary=True)

        for t in active_indices:
            y_t = y_display[mask, t]
            params.add(name=f"amp_t{t}", value=float(y_t.max() - y_t.min()), min=0)

        try:
            max_nfev = int(tb_nfev.text)
            if max_nfev <= 0:
                raise ValueError
        except ValueError:
            max_nfev = None

        try:
            weight_exp = float(tb_wexp.text)
        except ValueError:
            weight_exp = 0.0
            tb_wexp.set_val("0")

        # Per-trace weights: w_t = max(y_t)^weight_exp
        # sigma passed to lmfit = 1/w_t, repeated for every point in that trace.
        # weight_exp=0 → all ones (uniform); weight_exp=1 → tall traces weighted most.
        n_pts = len(x_sub)
        if weight_exp == 0.0:
            trace_weights = None
            print(f"Fitting… (max_iters={max_nfev or 'unlimited'}, weight_exp=0 — uniform)")
        else:
            trace_maxes   = np.maximum(np.max(y_sub_active, axis=0), 1e-10)
            trace_weights = trace_maxes ** weight_exp   # (n_active_traces,)
            # Normalise so the largest weight = 1 (keeps residual scale stable)
            trace_weights = trace_weights / trace_weights.max()
            print(f"Fitting… (max_iters={max_nfev or 'unlimited'}, weight_exp={weight_exp:.2f})")
            print(f"  Trace weights (normalised max^α): "
                  f"min={trace_weights.min():.3f}  max={trace_weights.max():.3f}")

        result = minimize(residual_global, params,
                          args=(x_sub, y_sub_active, n_peaks, active_indices,
                                trace_weights),
                          method='leastsq', max_nfev=max_nfev)

        print(f"Done.  success={result.success}  nfev={result.nfev}  "
              f"chisqr={result.chisqr:.2e}  reduced_chisq={result.redchi:.2e}")

        fp = result.params
        bl = fp["baseline"].value

        for i in range(n_peaks):
            prefix = f"p{i}_"
            print(f"  Peak {i+1}: center={fp[prefix+'center'].value:.5f}  "
                  f"sigma={fp[prefix+'sigma'].value:.5f}")
        print(f"  Baseline: {bl:.5e}")

        # ---- build fitted curves -------------------------------------------
        peak_sum     = np.zeros_like(x_sub)
        for i in range(n_peaks):
            prefix = f"p{i}_"
            model  = LorentzianModel(prefix=prefix)
            tp = Parameters()
            tp.add(name=prefix + 'amplitude', value=1.0)
            tp.add(name=prefix + 'center',    value=fp[prefix + "center"].value)
            tp.add(name=prefix + 'sigma',     value=fp[prefix + "sigma"].value)
            peak_sum += model.eval(tp, x=x_sub)

        y_fit = np.zeros((len(x_sub), n_traces))
        for i, t in enumerate(active_indices):
            y_fit[:, i] = fp[f"amp_t{t}"].value * peak_sum + bl

        # Update fit lines
        for i, t in enumerate(disp_idx):
            if trace_mask[t]:
                pos = np.where(active_indices == t)[0]
                if len(pos) > 0:
                    fit_lines[i].set_data(x_sub, y_fit[:, pos[0]])
            else:
                fit_lines[i].set_data([], [])

        # Component overlays
        if hasattr(fig, '_comp_lines'):
            for ln in fig._comp_lines:
                try:
                    ln.remove()
                except Exception:
                    pass
        fig._comp_lines = []

        t_ex = active_indices[int(np.argmax(np.max(y_sub_active, axis=0)))]
        for i in range(n_peaks):
            prefix = f"p{i}_"
            model  = LorentzianModel(prefix=prefix)
            tp = Parameters()
            tp.add(name=prefix + 'amplitude', value=fp[f"amp_t{t_ex}"].value)
            tp.add(name=prefix + 'center',    value=fp[prefix + "center"].value)
            tp.add(name=prefix + 'sigma',     value=fp[prefix + "sigma"].value)
            y_comp = model.eval(tp, x=x_sub) + bl
            ln, = ax.plot(x_sub, y_comp, '--', lw=1.5, alpha=0.7,
                          color=roi_color, label=f"Peak {i+1}")
            fig._comp_lines.append(ln)

        ln, = ax.plot(x_sub, bl * np.ones_like(x_sub), ':',
                      lw=2, color='k', label='Baseline')
        fig._comp_lines.append(ln)

        # Draggable center lines
        fitted_centers = [fp[f"p{i}_center"].value for i in range(n_peaks)]
        _make_center_lines(fitted_centers,
                           preserve_overrides=(center_init_mode == "pinned"))

        ax.relim()
        ax.autoscale_view()
        h, lb = ax.get_legend_handles_labels()
        ax.legend(dict(zip(lb, h)).values(), dict(zip(lb, h)).keys(),
                  fontsize=8, ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
        fig.canvas.draw_idle()

        # Save
        window_state["fitted_params"] = fp
        window_state["n_peaks"]       = n_peaks
        window_state["chisqr"]        = result.chisqr
        window_state["redchi"]        = result.redchi
        window_state["baseline_val"]  = bl

    btn_fresh.on_clicked(lambda e: _run_fit("auto"))
    btn_pinned.on_clicked(
        lambda e: _run_fit("auto") if len(center_lines) == 0
        else _run_fit("pinned")
    )

    widgets = dict(s_left=s_left, s_right=s_right, tb_n=tb_n,
                   tb_nfev=tb_nfev, btn_fresh=btn_fresh, btn_pinned=btn_pinned,
                   center_lines=center_lines, window_state=window_state)

    return fig, ax, window_state, widgets


# ---------------------------------------------------------------------------
# Window 1 — Background fitter
# ---------------------------------------------------------------------------
def interactive_background_fitter(x_data, y_data_all, label,
                                   prominence_factor=0.05,
                                   init_bounds=None, seed=101,
                                   real_times=None, trace_mask=None,
                                   baseline=None):
    if x_data[0] > x_data[-1]:
        x_data     = x_data[::-1]
        y_data_all = y_data_all[::-1, :]

    n_traces_total = y_data_all.shape[1]
    if trace_mask is None:
        trace_mask = np.ones(n_traces_total, dtype=bool)
    else:
        trace_mask = np.array(trace_mask, dtype=bool)
    active_indices = np.where(trace_mask)[0]

    accumulated_background = None
    pass_number = 1

    while True:
        if accumulated_background is not None:
            y_residual = y_data_all - accumulated_background
        else:
            y_residual = y_data_all

        result_holder = {"y_background": None, "bkg_state": {}, "do_again": False}

        def _extra(fig, ax):
            ax_skip  = plt.axes([0.60, 0.06, 0.18, 0.04])
            ax_again = plt.axes([0.60, 0.01, 0.18, 0.04])
            btn_skip  = Button(ax_skip,  "Skip (no background)")
            btn_again = Button(ax_again, "Correct Again")

            def on_skip(e):
                print("Background fitting skipped.")
                plt.close(fig)

            def on_again(e):
                ws  = widgets["window_state"]
                fp  = ws.get("fitted_params")
                n_p = ws.get("n_peaks", 0)
                if fp is not None and n_p > 0:
                    result_holder["do_again"] = True
                    plt.close(fig)
                else:
                    print("No fit recorded for this pass — run a fit before correcting again.")

            btn_skip.on_clicked(on_skip)
            btn_again.on_clicked(on_again)

            fig._btn_skip     = btn_skip
            fig._btn_again    = btn_again
            fig._ax_btn_skip  = ax_skip
            fig._ax_btn_again = ax_again

        fig, ax, window_state, widgets = _make_fitter_window(
            x_data=x_data,
            y_display=y_residual,
            label=(f"{label}\n"
                   f"Pass {pass_number}"
                   + (" [residual after previous correction]" if pass_number > 1 else "")
                   + " — fit background peaks, then close or correct again."),
            active_indices=active_indices,
            trace_mask=trace_mask,
            n_traces_total=n_traces_total,
            baseline=baseline,
            prominence_factor=prominence_factor,
            init_bounds=init_bounds,
            seed=seed + pass_number,
            real_times=real_times,
            n_peaks_label="n_bkg_peaks",
            roi_color="orange",
            extra_buttons_factory=_extra,
        )

        def on_close(event):
            ws  = widgets["window_state"]
            fp  = ws.get("fitted_params")
            n_p = ws.get("n_peaks", 0)
            if fp is not None and n_p > 0:
                y_this_pass = _eval_lorentzians(
                    fp, x_data, n_p, active_indices,
                    n_traces_total, trace_mask
                )
                result_holder["y_background"] = y_this_pass
                result_holder["bkg_state"] = {
                    "n_bkg_peaks":  n_p,
                    "bkg_roi":      (ws["lower_ppm_bound"], ws["upper_ppm_bound"]),
                    "chisqr":       ws.get("chisqr"),
                    "redchi":       ws.get("redchi"),
                    "baseline_val": ws.get("baseline_val"),
                    "n_passes":     pass_number,
                }
                print(f"\nBackground pass {pass_number} captured ({n_p} peak(s)).")
            else:
                print(f"\nNo fit recorded for pass {pass_number}.")

        fig.canvas.mpl_connect("close_event", on_close)
        plt.show()  # blocks until window is closed

        # Accumulate this pass's result
        if result_holder["y_background"] is not None:
            accumulated_background = (
                result_holder["y_background"] if accumulated_background is None
                else accumulated_background + result_holder["y_background"]
            )
            last_bkg_state = result_holder["bkg_state"]
        else:
            if pass_number == 1:
                last_bkg_state = {}

        # Loop again or exit
        if result_holder["do_again"]:
            pass_number += 1
            continue
        else:
            break

    if accumulated_background is not None:
        last_bkg_state["n_passes"] = pass_number
        print(f"\nTotal background passes: {pass_number}")

    return accumulated_background, last_bkg_state if accumulated_background is not None else {}


# ---------------------------------------------------------------------------
# Window 2 — Signal fitter
# ---------------------------------------------------------------------------

def interactive_peak_selector_global(x_data, y_data_all, ref_ppm, label,
                                     prominence_factor=0.05, base_fit_window=0.04,
                                     area_scaling_factor=1.0,
                                     init_bounds=None, seed=101, savepath=None,
                                     trace_indices=None, real_times=None,
                                     trace_mask=None, baseline=None,
                                     y_background=None):
    """
    Interactive global NMR signal fitter (Window 2).

    No n_bkg_peaks field — background is handled entirely by Window 1.
    If y_background is provided, it is subtracted from y_data_all before
    display and fitting.

    Parameters
    ----------
    baseline : float or None
        Fixed baseline from edit_baseline(); free if None.
    y_background : ndarray (N, T) or None
        Per-trace background profile from interactive_background_fitter().
    """
    if x_data[0] > x_data[-1]:
        x_data     = x_data[::-1]
        y_data_all = y_data_all[::-1, :]
        if y_background is not None:
            y_background = y_background[::-1, :]

    n_traces_total = y_data_all.shape[1]

    if y_background is not None:
        if y_background.shape != y_data_all.shape:
            raise ValueError(
                f"y_background shape {y_background.shape} != "
                f"y_data_all shape {y_data_all.shape}"
            )
        y_display = y_data_all - y_background
        print("\nBackground subtraction applied.")
    else:
        y_display = y_data_all

    if trace_mask is None:
        trace_mask = np.ones(n_traces_total, dtype=bool)
    else:
        trace_mask = np.array(trace_mask, dtype=bool)
    active_indices = np.where(trace_mask)[0]
    n_traces       = len(active_indices)

    print(f"Trace masking: {n_traces}/{n_traces_total} active")

    bkg_note = " [background subtracted]" if y_background is not None else ""

    fig, ax, window_state, widgets = _make_fitter_window(
        x_data=x_data,
        y_display=y_display,
        label=f"{label}{bkg_note} — {n_traces}/{n_traces_total} active traces",
        active_indices=active_indices,
        trace_mask=trace_mask,
        n_traces_total=n_traces_total,
        baseline=baseline,
        prominence_factor=prominence_factor,
        init_bounds=init_bounds,
        seed=seed,
        real_times=real_times,
        n_peaks_label="n_peaks",
        roi_color="green",
    )

    # ---- area calculation on close ----------------------------------------
    outer_window_state = {"seed": seed}
    if init_bounds is not None:
        outer_window_state["lower_ppm_bound"] = init_bounds[0]
        outer_window_state["upper_ppm_bound"] = init_bounds[1]
    else:
        outer_window_state["lower_ppm_bound"] = float(np.min(x_data))
        outer_window_state["upper_ppm_bound"] = float(np.max(x_data))

    def on_close(event):
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=300)
            print(f"Figure saved to {savepath}")

        ws = widgets["window_state"]
        fp = ws.get("fitted_params")
        if fp is None:
            print("Window closed without a completed fit — no areas computed.")
            return

        n_peaks  = ws["n_peaks"]
        left     = ws["lower_ppm_bound"]
        right    = ws["upper_ppm_bound"]
        bl       = ws["baseline_val"]

        # Peaks whose center is within the ROI contribute to area
        signal_indices = [i for i in range(n_peaks)
                          if left <= fp[f"p{i}_center"].value <= right]
        excluded       = [i for i in range(n_peaks) if i not in signal_indices]

        print(f"\n{'='*80}")
        print(f"AREAS BY TRACE  (ROI [{left:.4f}, {right:.4f}] ppm)")
        print(f"  Signal peaks contributing: {[i+1 for i in signal_indices]}")
        if excluded:
            print(f"  Excluded (outside ROI):    {[i+1 for i in excluded]}")
        print(f"{'='*80}")

        inner_mask = (x_data >= left) & (x_data <= right)
        x_inner    = x_data[inner_mask]
        areas      = []

        for t in range(n_traces_total):
            if not trace_mask[t]:
                areas.append(0.0)
                tl = f"{real_times[t]:.1f}" if real_times is not None else str(t)
                print(f"  Trace {t:3d} (t={tl:>6s}): MASKED")
                continue

            peak_area = 0.0
            for i in signal_indices:
                prefix = f"p{i}_"
                model  = LorentzianModel(prefix=prefix)
                tp = Parameters()
                tp.add(name=prefix + 'amplitude', value=fp[f"amp_t{t}"].value)
                tp.add(name=prefix + 'center',    value=fp[prefix + "center"].value)
                tp.add(name=prefix + 'sigma',     value=fp[prefix + "sigma"].value)
                peak_area += np.trapz(model.eval(tp, x=x_inner), x_inner)

            areas.append(peak_area)
            tl = f"{real_times[t]:.1f}" if real_times is not None else str(t)
            print(f"  Trace {t:3d} (t={tl:>6s}): {peak_area:12.4e}  "
                  f"(baseline={bl:.4e} excluded)")

        outer_window_state.update({
            "lower_ppm_bound": left,
            "upper_ppm_bound": right,
            "background_subtracted": y_background is not None,
            "component_params": [
                {
                    "peak_index":          i,
                    "in_roi":              left <= fp[f"p{i}_center"].value <= right,
                    "contributes_to_area": i in signal_indices,
                    "center":              fp[f"p{i}_center"].value,
                    "sigma":               fp[f"p{i}_sigma"].value,
                }
                for i in range(n_peaks)
            ],
            "signal_peak_indices":  signal_indices,
            "amplitudes_by_trace": [
                [fp[f"amp_t{t}"].value for _ in range(n_peaks)]
                if trace_mask[t] else [None] * n_peaks
                for t in range(n_traces_total)
            ],
            "baseline":  bl,
            "areas":     areas,
            "chisqr":    ws.get("chisqr"),
            "redchi":    ws.get("redchi"),
            "n_peaks":   n_peaks,
            "n_traces":  n_traces,
            "n_traces_total": n_traces_total,
        })

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()

    return outer_window_state