import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d


def edit_ppm_shifts(x, y_all_raw, label="PPM Shift Editor",
                    max_shift_ppm=10.0,
                    initial_shifts=None,
                    smooth_sigma=2):
    """
    Interactive editor for per-trace ppm shifts with shape-preserving auto-alignment.

    Parameters
    ----------
    x : (N,) array
        Common ppm axis
    y_all_raw : (N, T) array
        All FIDs / spectra (raw)
    label : str
        Figure title
    max_shift_ppm : float
        Maximum automatic shift allowed
    initial_shifts : (T,) array or None
        Initial per-trace shifts
    smooth_sigma : float
        Gaussian smoothing sigma for alignment only

    Returns
    -------
    shifts : (T,) array
        Final ppm shifts per trace
    """
    y_all_smooth = np.array([gaussian_filter1d(y, sigma=smooth_sigma)
                              for y in y_all_raw.T]).T
    n_traces = y_all_smooth.shape[1]
    shifts   = np.zeros(n_traces) if initial_shifts is None else initial_shifts.copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(label)
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()

    lines = []
    for t in range(n_traces):
        line, = ax.plot(x + shifts[t], y_all_smooth[:, t],
                        lw=1, alpha=0.35, picker=5)
        lines.append(line)

    active = {"trace": None, "last_x": None}

    def redraw():
        for t in range(n_traces):
            lines[t].set_xdata(x + shifts[t])
        fig.canvas.draw_idle()

    # Original version
    # def infer_shifts_peakmax():
    #     ref_idx      = np.argmax(y_all_smooth.max(axis=0))
    #     ref          = y_all_smooth[:, ref_idx]
    #     ref_peak_x   = x[np.argmax(ref)]
    #     new_shifts   = np.zeros(n_traces)
    #     for t in range(n_traces):
    #         sig_peak_x  = x[np.argmax(y_all_smooth[:, t])]
    #         shift       = ref_peak_x - sig_peak_x
    #         new_shifts[t] = np.clip(shift, -max_shift_ppm, max_shift_ppm)
    #     return new_shifts

    def infer_shifts_peakmax():
        hi, lo = ax.get_xlim()
        mask   = (x >= lo) & (x <= hi)
        x_win  = x[mask]
        y_win  = y_all_smooth[mask, :]

        print(f"Window: {lo:.4f} – {hi:.4f} ppm,  {mask.sum()} points selected")
        print(f"y_win shape: {y_win.shape}")

        ref_idx    = np.argmax(y_win.max(axis=0))
        ref_peak_x = x_win[np.argmax(y_win[:, ref_idx])]
        print(f"Reference trace: {ref_idx},  ref peak @ {ref_peak_x:.4f} ppm")

        new_shifts = np.zeros(n_traces)
        for t in range(n_traces):
            sig_peak_x    = x_win[np.argmax(y_win[:, t])]
            shift         = ref_peak_x - sig_peak_x
            print(f"  trace {t}: peak @ {sig_peak_x:.4f},  shift = {shift:+.5f}")
            new_shifts[t] = np.clip(shift, -max_shift_ppm, max_shift_ppm)
        return new_shifts


    def on_press(event):
        if event.inaxes != ax:
            return
        for t, line in enumerate(lines):
            hit, _ = line.contains(event)
            if hit:
                active["trace"]  = t
                active["last_x"] = event.xdata
                for l in lines:
                    l.set_alpha(0.15)
                line.set_alpha(1.0)
                break

    def on_motion(event):
        t = active["trace"]
        if t is None or event.xdata is None:
            return
        shifts[t]   += event.xdata - active["last_x"]
        active["last_x"] = event.xdata
        lines[t].set_xdata(x + shifts[t])
        fig.canvas.draw_idle()

    def on_release(event):
        active["trace"]  = None
        active["last_x"] = None
        for l in lines:
            l.set_alpha(0.35)
        print("Shifts (ppm):", shifts)
        fig.canvas.draw_idle()

    def on_key(event):
        if active["trace"] is None:
            return
        t    = active["trace"]
        step = 0.0002
        if event.key == "left":
            shifts[t] -= step
        elif event.key == "right":
            shifts[t] += step
        lines[t].set_xdata(x + shifts[t])
        fig.canvas.draw_idle()

    def on_auto(event):
        nonlocal shifts
        shifts[:] = infer_shifts_peakmax()
        print("Auto shifts (ppm):", shifts)
        redraw()

    def on_reset(event):
        nonlocal shifts
        shifts[:] = 0.0
        redraw()
        print("All ppm shifts reset to 0")

    ax_btn       = plt.axes([0.82, 0.02, 0.15, 0.05])
    btn          = Button(ax_btn, "Auto-align")
    ax_btn_reset = plt.axes([0.65, 0.02, 0.15, 0.05])
    btn_reset    = Button(ax_btn_reset, "Reset shifts")

    btn.on_clicked(on_auto)
    btn_reset.on_clicked(on_reset)

    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event",      on_key)

    plt.show()
    return shifts


def apply_ppm_shifts(x, y_all, ppm_shifts):
    y_aligned = np.zeros_like(y_all)
    for t, shift in enumerate(ppm_shifts):
        y_aligned[:, t] = np.interp(
            x,
            x + shift,
            y_all[:, t],
            left=0.0,
            right=0.0
        )
    return y_aligned


def edit_baseline(x, y_aligned, label="Baseline Editor",
                  percentile=10):
    """
    Interactive baseline inspector/editor.

    Estimates the baseline as the `percentile`-th percentile of ALL points
    across ALL traces in the window, then displays a draggable horizontal
    dotted line so the user can adjust it if needed.

    Parameters
    ----------
    x : (N,) array
        Common ppm axis (the fitting window, already trimmed)
    y_aligned : (N, T) array
        Aligned spectra
    label : str
        Figure title
    percentile : float
        Percentile used for the initial estimate (default 10)

    Returns
    -------
    baseline : float
        The (possibly user-adjusted) baseline value
    """
    # ---- initial estimate --------------------------------------------------
    baseline_est = float(np.percentile(y_aligned, percentile))
    state        = {"baseline": baseline_est}

    print(f"\nBaseline estimate ({percentile}th percentile): {baseline_est:.5e}")

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    plt.subplots_adjust(bottom=0.18)
    ax.set_title(f"{label}\n"
                 f"Drag the dotted line to adjust baseline  "
                 f"(initial = {baseline_est:.4e}, {percentile}th pct)")
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()

    n_traces = y_aligned.shape[1]
    colors   = plt.cm.viridis(np.linspace(0, 1, n_traces))
    for t in range(n_traces):
        ax.plot(x, y_aligned[:, t], lw=1, alpha=0.35, color=colors[t])

    # Draggable baseline
    bl_line, = ax.plot(x, np.full_like(x, baseline_est),
                       'k--', lw=2.5, alpha=0.85,
                       label=f"Baseline = {baseline_est:.4e}")
    ax.legend(fontsize=9, loc="upper right")

    drag = {"active": False, "last_y": None}

    def _update_label():
        bl_line.set_label(f"Baseline = {state['baseline']:.4e}")
        ax.legend(fontsize=9, loc="upper right")
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return
        hit, _ = bl_line.contains(event)
        if hit:
            drag["active"] = True
            drag["last_y"] = event.ydata
            bl_line.set_linewidth(4)
            fig.canvas.draw_idle()

    def on_motion(event):
        if not drag["active"] or event.ydata is None:
            return
        dy = event.ydata - drag["last_y"]
        state["baseline"] += dy
        drag["last_y"]     = event.ydata
        bl_line.set_ydata(np.full_like(x, state["baseline"]))
        _update_label()

    def on_release(event):
        if drag["active"]:
            drag["active"] = False
            bl_line.set_linewidth(2.5)
            print(f"Baseline set to: {state['baseline']:.5e}")
            _update_label()

    # Reset button
    ax_btn_reset = plt.axes([0.15, 0.04, 0.18, 0.06])
    btn_reset    = Button(ax_btn_reset, "Reset to estimate")

    def on_reset(event):
        state["baseline"] = baseline_est
        bl_line.set_ydata(np.full_like(x, baseline_est))
        print(f"Baseline reset to estimate: {baseline_est:.5e}")
        _update_label()

    btn_reset.on_clicked(on_reset)

    fig.canvas.mpl_connect("button_press_event",   on_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    plt.show()

    print(f"Final baseline: {state['baseline']:.5e}")
    return state["baseline"]