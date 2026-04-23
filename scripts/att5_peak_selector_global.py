import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
from lmfit import Parameters, minimize
from lmfit.models import LorentzianModel, ConstantModel
from scipy.signal import find_peaks

def global_model_lmfit(params, x_data, n_peaks, n_traces):
    result = np.zeros((len(x_data), n_traces))
    
    for i in range(n_peaks):
        prefix = f"p{i}_"
        center = params[prefix + "center"].value
        sigma = params[prefix + "sigma"].value
        
        model = LorentzianModel(prefix=prefix)
        
        for t in range(n_traces):            
            amp = params[f"{prefix}amp_t{t}"].value
            
            temp_params = Parameters()
            temp_params.add(prefix + 'amplitude', value=amp)
            temp_params.add(prefix + 'center', value=center)
            temp_params.add(prefix + 'sigma', value=sigma)
            
            result[:, t] += model.eval(temp_params, x=x_data)
    
    # no baseline
    # for t in range(n_traces):
    #     result[:, t] += params[f"bkg_t{t}"].value
    
    return result

def global_model_lmfit_scaled(params, x_data, n_peaks, n_traces):
    """
    Peaks: shared parameters (center, sigma) across all FIDs
    Amplitude: one scaling per FID applied to the sum of all peaks
    Baseline: single global baseline applied to all traces
    """
    result = np.zeros((len(x_data), n_traces))
    
    # Compute sum of all peaks (unscaled)
    peak_sum = np.zeros_like(x_data)
    for i in range(n_peaks):
        prefix = f"p{i}_"
        center = params[prefix + "center"].value
        sigma  = params[prefix + "sigma"].value
        
        model = LorentzianModel(prefix=prefix)
        temp_params = Parameters()
        temp_params.add(prefix + 'amplitude', value=1.0)  # unit amplitude
        temp_params.add(prefix + 'center', value=center)
        temp_params.add(prefix + 'sigma', value=sigma)
        
        peak_sum += model.eval(temp_params, x=x_data)
    
    # Get global baseline
    baseline = params["baseline"].value
    
    # Scale sum by per-trace amplitude and add baseline
    for t in range(n_traces):
        amp_scale = params[f"amp_t{t}"].value
        result[:, t] = amp_scale * peak_sum + baseline
    
    return result


def residual_global(params, x_data, y_data, n_peaks, trace_indices):
    """
    Residual function for global fit
    
    Parameters
    ----------
    trace_indices : array-like
        Indices of traces to include in the fit
    """
    n_traces = len(trace_indices)
    result = np.zeros((len(x_data), n_traces))
    
    # Compute sum of all peaks (unscaled)
    peak_sum = np.zeros_like(x_data)
    for i in range(n_peaks):
        prefix = f"p{i}_"
        center = params[prefix + "center"].value
        sigma  = params[prefix + "sigma"].value
        
        model = LorentzianModel(prefix=prefix)
        temp_params = Parameters()
        temp_params.add(prefix + 'amplitude', value=1.0)
        temp_params.add(prefix + 'center', value=center)
        temp_params.add(prefix + 'sigma', value=sigma)
        
        peak_sum += model.eval(temp_params, x=x_data)
    
    # Get global baseline
    baseline = params["baseline"].value
    
    # Scale sum by per-trace amplitude and add baseline
    for i, t in enumerate(trace_indices):
        amp_scale = params[f"amp_t{t}"].value
        result[:, i] = amp_scale * peak_sum + baseline
    
    return (y_data - result).ravel()

def interactive_peak_selector_global(x_data, y_data_all, ref_ppm, label,
                                     prominence_factor=0.05, base_fit_window=0.04,
                                     area_scaling_factor=1.0,
                                     init_bounds=None, seed=101, savepath=None,
                                     trace_indices=None, real_times=None, trace_mask=None):
    """
    Interactive viewer with global fitting across all FIDs.
    Uses lmfit's LorentzianModel with a single global baseline.
    
    Parameters
    ----------
    trace_mask : array-like of bool, optional
        Boolean mask of shape (n_traces,). True = include in fit, False = exclude.
        Excluded traces will have their areas reported as 0.
    """
    
    # Ensure ascending x for fitting
    if x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data_all = y_data_all[::-1, :]
    
    n_traces_total = y_data_all.shape[1]
    
    # Create trace mask if not provided (default: include all)
    if trace_mask is None:
        trace_mask = np.ones(n_traces_total, dtype=bool)
    else:
        trace_mask = np.array(trace_mask, dtype=bool)
        if len(trace_mask) != n_traces_total:
            raise ValueError(f"trace_mask length ({len(trace_mask)}) must match number of traces ({n_traces_total})")
    
    # Separate active traces (for fitting) from all traces
    active_indices = np.where(trace_mask)[0]
    n_traces = len(active_indices)  # Number of traces to actually fit
    
    print(f"\nTrace masking: {n_traces} active traces out of {n_traces_total} total")
    print(f"Active trace indices: {active_indices.tolist()}")
    
    if trace_indices is None:
        trace_indices = list(range(n_traces_total))
    
    # Initial plot
    fig, ax_main = plt.subplots(1, 1, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.30, left=0.1, right=0.85)

    
    ax_main.set_title(f"{label} - Global Fit: {n_traces}/{n_traces_total} Active Traces")
    ax_main.set_xlabel("ppm")
    ax_main.set_ylabel("Intensity")
    ax_main.invert_xaxis()
    
    # Plot a subset of traces for clarity (prefer active traces)
    n_display = min(10, n_traces_total)
    
    # Prioritize showing active traces
    if n_traces <= n_display:
        # Show all active traces plus some inactive ones
        display_indices = list(active_indices)
        inactive_indices = np.where(~trace_mask)[0]
        remaining_slots = n_display - n_traces
        if remaining_slots > 0 and len(inactive_indices) > 0:
            n_inactive_show = min(remaining_slots, len(inactive_indices))
            inactive_show = np.linspace(0, len(inactive_indices)-1, n_inactive_show, dtype=int)
            display_indices.extend(inactive_indices[inactive_show].tolist())
        display_indices = np.array(display_indices)
    else:
        # Show subset of active traces
        display_indices = active_indices[np.linspace(0, n_traces-1, n_display, dtype=int)]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(display_indices)))
    
    data_lines = []
    fit_lines = []
    
    for i, t in enumerate(display_indices):
        is_active = trace_mask[t]
        alpha = 0.6 if is_active else 0.2
        marker_style = '.' if is_active else 'x'
        label_suffix = "" if is_active else " (masked)"
        
        line_data, = ax_main.plot(x_data, y_data_all[:, t], marker_style, 
                                   alpha=alpha, color=colors[i], markersize=3,
                                   label=f"Trace {t}{label_suffix}")
        line_fit, = ax_main.plot([], [], '-', color=colors[i], lw=2, alpha=alpha)
        
        data_lines.append(line_data)
        fit_lines.append(line_fit)
    
    ax_main.legend(fontsize=8, ncol=2, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Sliders for region selection
    axcolor = "lightgoldenrodyellow"
    ax_left = plt.axes([0.15, 0.15, 0.65, 0.02], facecolor=axcolor)
    ax_right = plt.axes([0.15, 0.11, 0.65, 0.02], facecolor=axcolor)
    
    window_state = {"seed": seed}
    if init_bounds is None:
        window_state["lower_ppm_bound"] = np.min(x_data)
        window_state["upper_ppm_bound"] = np.max(x_data)
    else:
        window_state["lower_ppm_bound"], window_state["upper_ppm_bound"] = init_bounds
    
    s_left = Slider(ax_left, "Lower PPM", np.min(x_data), np.max(x_data),
                    valinit=window_state["lower_ppm_bound"])
    s_right = Slider(ax_right, "Upper PPM", np.min(x_data), np.max(x_data),
                     valinit=window_state["upper_ppm_bound"])
    
    vline_left = ax_main.axvline(s_left.val, color="g", linestyle="--", lw=2)
    vline_right = ax_main.axvline(s_right.val, color="g", linestyle="--", lw=2)
    
    def update_lines(val=None):
        left, right = sorted([s_left.val, s_right.val])
        vline_left.set_xdata([left, left])
        vline_right.set_xdata([right, right])
        fig.canvas.draw_idle()
    
    s_left.on_changed(update_lines)
    s_right.on_changed(update_lines)
    
    # Button and TextBox
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.04])
    btn = Button(ax_button, "Refit All FIDs")
    
    ax_text = plt.axes([0.15, 0.02, 0.15, 0.04])
    text_box = TextBox(ax_text, "n_peaks", initial="")
    
    def on_button(event):
        window_state["seed"] += 1
        np.random.seed(window_state["seed"])
        print(f"\n{'='*80}")
        print(f"GLOBAL FIT - Random seed: {window_state['seed']}")
        print(f"{'='*80}\n")
        
        left, right = sorted([s_left.val, s_right.val])
        outer_left, outer_right = left - 0.05, right + 0.05
        window_state["lower_ppm_bound"] = left
        window_state["upper_ppm_bound"] = right
        
        mask = (x_data >= outer_left) & (x_data <= outer_right)
        if np.sum(mask) < 5:
            print("Too few points in region")
            return
        
        x_sub = x_data[mask]
        y_sub = y_data_all[mask, :]
        
        # Extract only active traces for fitting
        y_sub_active = y_sub[:, active_indices]
        
        # Peak detection on the mean spectrum of ACTIVE traces only
        y_mean = np.mean(y_sub_active, axis=1)
        peaks_idx, _ = find_peaks(
            y_mean, prominence=prominence_factor * (y_mean.max() - y_mean.min())
        )
        if len(peaks_idx) == 0:
            peaks_idx = [np.argmax(y_mean)]
        
        max_num_peaks = 10
        if len(peaks_idx) > max_num_peaks:
            distances = np.abs(x_sub[peaks_idx] - ref_ppm)
            closest_idx = np.argsort(distances)[:max_num_peaks]
            peaks_idx = [peaks_idx[i] for i in closest_idx]
        
        # Get n_peaks from TextBox
        try:
            n_peaks = int(text_box.text)
        except ValueError:
            n_peaks = len(peaks_idx)
            text_box.set_val(str(n_peaks))
        
        if n_peaks == 1:
            n_peaks += 1
        
        print(f"Fitting {n_peaks} peaks across {n_traces} FIDs")
        print(f"Region: [{left:.4f}, {right:.4f}] ppm")
        
        # Build global parameters
        params = Parameters()
        
        # Shared peak parameters
        center_var = 0.05
        for i in range(n_peaks):
            prefix = f"p{i}_"
            if i < len(peaks_idx):
                peak_idx = peaks_idx[i]
                center_guess = x_sub[peak_idx] + np.random.uniform(-center_var, center_var)
            else:
                center_guess = ref_ppm + np.random.uniform(-0.03, 0.03)

            params.add(prefix + "center", value=center_guess,
                    min=x_sub.min(), max=x_sub.max())
            params.add(prefix + "sigma", value=0.005 * np.random.uniform(0.8, 1.2),
                    min=0.001, max=0.03)

        # Single global baseline
        baseline_guess = np.min(y_sub_active)
        params.add("baseline", value=baseline_guess)

        # Per-FID amplitude (only for active traces)
        for t in active_indices:
            y_t = y_sub[:, t]
            amp_guess = (y_t.max() - y_t.min())  # rough guess for scaling
            params.add(f"amp_t{t}", value=amp_guess, min=0)
        
        # Fit peaks globally (only active traces)
        print("Fitting... (this may take a moment)")
        result = minimize(residual_global, params, 
                         args=(x_sub, y_sub_active, n_peaks, active_indices),
                         method='leastsq')
        
        print(f"\nFit complete!")
        print(f"Success: {result.success}")
        print(f"Chi-square: {result.chisqr:.2e}")
        print(f"Reduced Chi-square: {result.redchi:.2e}")
        
        # Extract results
        fitted_params = result.params
        
        # Compute fitted values for active traces
        y_fit_active = np.zeros((len(x_sub), n_traces))
        peak_sum = np.zeros_like(x_sub)
        for i in range(n_peaks):
            prefix = f"p{i}_"
            center = fitted_params[prefix + "center"].value
            sigma = fitted_params[prefix + "sigma"].value
            
            model = LorentzianModel(prefix=prefix)
            temp_params = Parameters()
            temp_params.add(prefix + 'amplitude', value=1.0)
            temp_params.add(prefix + 'center', value=center)
            temp_params.add(prefix + 'sigma', value=sigma)
            
            peak_sum += model.eval(temp_params, x=x_sub)
        
        baseline = fitted_params["baseline"].value
        
        for i, t in enumerate(active_indices):
            amp_scale = fitted_params[f"amp_t{t}"].value
            y_fit_active[:, i] = amp_scale * peak_sum + baseline
        
        # Print shared peak parameters
        print(f"\n{'='*80}")
        print("SHARED PEAK PARAMETERS:")
        print(f"{'='*80}")
        for i in range(n_peaks):
            prefix = f"p{i}_"
            center = fitted_params[prefix + "center"].value
            sigma = fitted_params[prefix + "sigma"].value
            print(f"Peak {i+1}:")
            print(f"  Center: {center:.5f} ppm")
            print(f"  Sigma (width): {sigma:.5f}")
        
        # Print baseline
        baseline = fitted_params["baseline"].value
        print(f"\nGlobal Baseline: {baseline:.5e}")
        
        # Calculate areas for each trace
        print(f"\n{'='*80}")
        print("AREAS BY TRACE:")
        print(f"{'='*80}")
        
        inner_mask = (x_sub >= left) & (x_sub <= right)
        x_inner = x_sub[inner_mask]
        areas = []
        
        # Calculate for ALL traces (not just active ones)
        for t in range(n_traces_total):
            if not trace_mask[t]:
                # Masked trace - report zero area
                areas.append(0.0)
                time_label = f"{real_times[t]:.1f}" if real_times is not None else str(t)
                print(f"Trace {t:3d} (t={time_label:>6s}): MASKED - Area = 0.0")
                continue
            
            # Active trace - calculate area
            peak_area = 0.0
            for i in range(n_peaks):
                prefix = f"p{i}_"
                center = fitted_params[prefix + "center"].value
                
                if left <= center <= right:
                    amp = fitted_params[f"amp_t{t}"].value
                    sigma = fitted_params[prefix + "sigma"].value
                    
                    # Create temporary model and params for this peak
                    model = LorentzianModel(prefix=prefix)
                    temp_params = Parameters()
                    temp_params.add(prefix + 'amplitude', value=amp)
                    temp_params.add(prefix + 'center', value=center)
                    temp_params.add(prefix + 'sigma', value=sigma)
                    
                    # Evaluate and integrate
                    y_peak = model.eval(temp_params, x=x_inner)
                    peak_area += np.trapz(y_peak, x_inner)
            
            # Add baseline contribution if above zero
            baseline_area = 0.0
            if baseline > 0:
                # Rectangle: height * width
                baseline_area = baseline * (right - left)
            
            total_area = peak_area + baseline_area
            areas.append(total_area)
            
            time_label = f"{real_times[t]:.1f}" if real_times is not None else str(t)
            print(f"Trace {t:3d} (t={time_label:>6s}): Peak={peak_area:12.4e}, Baseline={baseline_area:12.4e}, Total={total_area:12.4e}")
        
        # Update plots (only for displayed traces)
        for i, t in enumerate(display_indices):
            if trace_mask[t]:
                # Find position in active_indices
                active_pos = np.where(active_indices == t)[0]
                if len(active_pos) > 0:
                    fit_lines[i].set_data(x_sub, y_fit_active[:, active_pos[0]])
            else:
                # Masked trace - no fit line
                fit_lines[i].set_data([], [])
        
        # Plot individual peak components on the first trace
        if hasattr(fig, 'component_lines'):
            for line in fig.component_lines:
                line.remove()
        fig.component_lines = []
        
        # Find trace with largest max intensity inside fitting window (among active traces)
        max_vals = np.max(y_sub_active, axis=0)
        t_example_pos = np.argmax(max_vals)
        t_example = active_indices[t_example_pos]
        
        for i in range(n_peaks):
            prefix = f"p{i}_"
            center = fitted_params[prefix + "center"].value
            sigma = fitted_params[prefix + "sigma"].value
            amp = fitted_params[f"amp_t{t_example}"].value
            
            # Use lmfit model for correct evaluation
            model = LorentzianModel(prefix=prefix)
            temp_params = Parameters()
            temp_params.add(prefix + 'amplitude', value=amp)
            temp_params.add(prefix + 'center', value=center)
            temp_params.add(prefix + 'sigma', value=sigma)
            
            y_comp = model.eval(temp_params, x=x_sub) + baseline
            line, = ax_main.plot(x_sub, y_comp, '--', lw=1.5, alpha=0.7,
                                label=f'Peak {i+1} (trace {t_example})')
            fig.component_lines.append(line)
        
        # Plot global baseline
        y_baseline = baseline * np.ones_like(x_sub)
        line, = ax_main.plot(x_sub, y_baseline, ':', lw=2, color='k',
                            label=f'Global Baseline')
        fig.component_lines.append(line)
        
        # Update axes
        ax_main.relim()
        ax_main.autoscale_view()
        
        # Update legend
        handles, labels = ax_main.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_main.legend(unique.values(), unique.keys(), fontsize=8, ncol=2,
                      loc='center left', bbox_to_anchor=(1, 0.5))
        
        fig.canvas.draw_idle()
        
        # Save results to window_state
        component_params = []
        for i in range(n_peaks):
            prefix = f"p{i}_"
            component_params.append({
                "center": fitted_params[prefix + "center"].value,
                "sigma": fitted_params[prefix + "sigma"].value,
            })
        
        amplitudes_by_trace = []
        
        # Save amplitudes for ALL traces (None for masked ones)
        for t in range(n_traces_total):
            if trace_mask[t]:
                amps = [fitted_params[f"amp_t{t}"].value for i in range(n_peaks)]
            else:
                amps = [None] * n_peaks  # Masked trace
            amplitudes_by_trace.append(amps)
        
        window_state["fit_results"] = {
            "x": x_sub,
            "y_fit_active": y_fit_active,
            "active_indices": active_indices.tolist(),
            "trace_mask": trace_mask.tolist(),
            "component_params": component_params,
            "amplitudes_by_trace": amplitudes_by_trace,
            "baseline": baseline,
            "areas": areas,
            "chisqr": result.chisqr,
            "redchi": result.redchi,
            "n_peaks": n_peaks,
            "n_traces": n_traces,
            "n_traces_total": n_traces_total
        }
    
    btn.on_clicked(on_button)
    
    def on_close(event):
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=300)
            print(f"\nFigure saved to {savepath}")
    
    fig.canvas.mpl_connect("close_event", on_close)
    
    plt.show()
    
    return window_state