import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.signal import correlate
from scipy.ndimage import gaussian_filter1d

def edit_ppm_shifts(x, y_all_raw, label="PPM Shift Editor",
                    max_shift_ppm=0.02,
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
    # ---------- prep ----------
    y_all_smooth = np.array([gaussian_filter1d(y, sigma=smooth_sigma) for y in y_all_raw.T]).T
    n_traces = y_all_smooth.shape[1]
    shifts = np.zeros(n_traces) if initial_shifts is None else initial_shifts.copy()

    # ---------- figure ----------
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

    # ---------- helpers ----------
    def redraw():
        for t in range(n_traces):
            lines[t].set_xdata(x + shifts[t])
        fig.canvas.draw_idle()

    # ---------- shape-preserving auto alignment ----------
    def infer_shifts_peakmax():
        """
        Align spectra by their tallest peak to the tallest peak of the reference.
        """
        # Reference: spectrum with tallest peak
        ref_idx = np.argmax(y_all_smooth.max(axis=0))
        ref = y_all_smooth[:, ref_idx]
        ref_peak_idx = np.argmax(ref)
        ref_peak_x = x[ref_peak_idx]

        new_shifts = np.zeros(n_traces)
        for t in range(n_traces):
            sig = y_all_smooth[:, t]
            sig_peak_idx = np.argmax(sig)
            sig_peak_x = x[sig_peak_idx]

            # shift to align peak maxima
            shift = ref_peak_x - sig_peak_x
            # clip to allowed max shift
            shift = np.clip(shift, -max_shift_ppm, max_shift_ppm)
            new_shifts[t] = shift

        return new_shifts


    # ---------- mouse interaction ----------
    def on_press(event):
        if event.inaxes != ax:
            return
        for t, line in enumerate(lines):
            hit, _ = line.contains(event)
            if hit:
                active["trace"] = t
                active["last_x"] = event.xdata
                for l in lines:
                    l.set_alpha(0.15)
                line.set_alpha(1.0)
                break

    def on_motion(event):
        t = active["trace"]
        if t is None or event.xdata is None:
            return
        dx_mouse = event.xdata - active["last_x"]
        shifts[t] += dx_mouse
        active["last_x"] = event.xdata
        lines[t].set_xdata(x + shifts[t])
        fig.canvas.draw_idle()

    def on_release(event):
        active["trace"] = None
        active["last_x"] = None
        for l in lines:
            l.set_alpha(0.35)
        print("Auto shifts (ppm):", shifts)
        fig.canvas.draw_idle()

    # ---------- keyboard nudging ----------
    def on_key(event):
        if active["trace"] is None:
            return
        t = active["trace"]
        step = 0.0002
        if event.key == "left":
            shifts[t] -= step
        elif event.key == "right":
            shifts[t] += step
        lines[t].set_xdata(x + shifts[t])
        fig.canvas.draw_idle()

    # ---------- buttons ----------
    ax_btn = plt.axes([0.82, 0.02, 0.15, 0.05])
    btn = Button(ax_btn, "Auto-align")

    def on_auto(event):
        nonlocal shifts
        # shifts[:] = infer_shifts()
        shifts[:] = infer_shifts_peakmax()
        print("Auto shifts (ppm):", shifts)
        redraw()
        print("Automatic ppm alignment applied")

    btn.on_clicked(on_auto)

    ax_btn_reset = plt.axes([0.65, 0.02, 0.15, 0.05])
    btn_reset = Button(ax_btn_reset, "Reset shifts")

    def on_reset(event):
        nonlocal shifts
        shifts[:] = 0.0
        redraw()
        print("All ppm shifts reset to 0")

    btn_reset.on_clicked(on_reset)

    # ---------- events ----------
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()

    return shifts



def apply_ppm_shifts(x, y_all, ppm_shifts):
    y_aligned = np.zeros_like(y_all)

    for t, shift in enumerate(ppm_shifts):
        y_aligned[:, t] = np.interp(
            x,
            x + shift,          # ← shifted domain (CRITICAL)
            y_all[:, t],
            left=0.0,
            right=0.0
        )

    return y_aligned

