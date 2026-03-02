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
    
    # Scale sum by per-trace amplitude
    for t in range(n_traces):
        amp_scale = params[f"amp_t{t}"].value
        result[:, t] = amp_scale * peak_sum
    
    return result


def residual_global(params, x_data, y_data, n_peaks, n_traces):
    """Residual function for global fit"""
    model = global_model_lmfit_scaled(params, x_data, n_peaks, n_traces)
    return (y_data - model).ravel()

def interactive_peak_selector_global(x_data, y_data_all, ref_ppm, label,
                                     prominence_factor=0.05, base_fit_window=0.04,
                                     area_scaling_factor=1.0,
                                     init_bounds=None, seed=101, savepath=None,
                                     trace_indices=None, real_times=None):
    """
    Interactive viewer with global fitting across all FIDs.
    Uses lmfit's PseudoVoigtModel for correct normalization.
    """
    
    # Ensure ascending x for fitting
    if x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data_all = y_data_all[::-1, :]
    
    n_traces = y_data_all.shape[1]
    if trace_indices is None:
        trace_indices = list(range(n_traces))
    
    # Initial plot
    fig, ax_main = plt.subplots(1, 1, figsize=(12, 6))
    plt.subplots_adjust(bottom=0.30, left=0.1, right=0.85)

    
    ax_main.set_title(f"{label} - Global Fit Across {n_traces} FIDs")
    ax_main.set_xlabel("ppm")
    ax_main.set_ylabel("Intensity")
    ax_main.invert_xaxis()
    
    # Plot a subset of traces for clarity
    n_display = min(10, n_traces)
    display_indices = np.linspace(0, n_traces-1, n_display, dtype=int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_display))
    
    data_lines = []
    fit_lines = []
    
    for i, t in enumerate(display_indices):
        line_data, = ax_main.plot(x_data, y_data_all[:, t], '.', 
                                   alpha=0.6, color=colors[i], markersize=3,
                                   label=f"Trace {t}")
        line_fit, = ax_main.plot([], [], '-', color=colors[i], lw=2)
        
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
        
        # Peak detection on the mean spectrum
        y_mean = np.mean(y_sub, axis=1)
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

        # Per-FID amplitude
        for t in range(n_traces):
            y_t = y_sub[:, t]
            amp_guess = (y_t.max() - y_t.min())  # rough guess for scaling
            params.add(f"amp_t{t}", value=amp_guess, min=0)
        
        # Fit peaks globally
        print("Fitting... (this may take a moment)")
        result = minimize(residual_global, params, 
                         args=(x_sub, y_sub, n_peaks, n_traces),
                         method='leastsq')
        
        print(f"\nFit complete!")
        print(f"Success: {result.success}")
        print(f"Chi-square: {result.chisqr:.2e}")
        print(f"Reduced Chi-square: {result.redchi:.2e}")
        
        # Extract results
        fitted_params = result.params
        y_fit = global_model_lmfit_scaled(fitted_params, x_sub, n_peaks, n_traces)
        
        # Print shared peak parameters
        print(f"\n{'='*80}")
        print("SHARED PEAK PARAMETERS:")
        print(f"{'='*80}")
        for i in range(n_peaks):
            prefix = f"p{i}_"
            center = fitted_params[prefix + "center"].value
            sigma = fitted_params[prefix + "sigma"].value
            # fraction = fitted_params[prefix + "fraction"].value
            print(f"Peak {i+1}:")
            print(f"  Center: {center:.5f} ppm")
            print(f"  Sigma (width): {sigma:.5f}")
            # print(f"  Fraction (Lorentzian): {fraction:.3f}")
        
        # Calculate areas for each trace
        # With lmfit's PseudoVoigt, the 'amplitude' parameter IS the area!
        print(f"\n{'='*80}")
        print("AREAS BY TRACE:")
        print(f"{'='*80}")
        
        inner_mask = (x_sub >= left) & (x_sub <= right)
        areas = []
        
        for t in range(n_traces):
            total_area = 0.0
            for i in range(n_peaks):
                prefix = f"p{i}_"
                center = fitted_params[prefix + "center"].value
                
                if left <= center <= right:
                    # For lmfit's PseudoVoigt, amplitude IS the area
                    # But we still need to integrate only the part within bounds
                    # amp = fitted_params[f"{prefix}amp_t{t}"].value
                    amp = fitted_params[f"amp_t{t}"].value
                    sigma = fitted_params[prefix + "sigma"].value
                    # fraction = fitted_params[prefix + "fraction"].value
                    
                    # Create temporary model and params for this peak
                    model = LorentzianModel(prefix=prefix)
                    temp_params = Parameters()
                    temp_params.add(prefix + 'amplitude', value=amp)
                    temp_params.add(prefix + 'center', value=center)
                    temp_params.add(prefix + 'sigma', value=sigma)
                    # temp_params.add(prefix + 'fraction', value=fraction)
                    
                    # Evaluate and integrate
                    y_peak = model.eval(temp_params, x=x_sub[inner_mask])
                    total_area += np.trapz(y_peak, x_sub[inner_mask])
            
            areas.append(total_area)
            time_label = f"{real_times[t]:.1f}" if real_times is not None else str(t)
            print(f"Trace {t:3d} (t={time_label:>6s}): {total_area:12.4e}")
        
        # Update plots
        for i, t in enumerate(display_indices):
            fit_lines[i].set_data(x_sub, y_fit[:, t])
        
        # Plot individual peak components on the first trace
        if hasattr(fig, 'component_lines'):
            for line in fig.component_lines:
                line.remove()
        fig.component_lines = []
        
        # Find trace with largest max intensity inside fitting window
        # t_example = display_indices[0]
        max_vals = np.max(y_sub, axis=0)   # shape: (n_traces,)
        t_example = np.argmax(max_vals)
        
        for i in range(n_peaks):
            prefix = f"p{i}_"
            center = fitted_params[prefix + "center"].value
            sigma = fitted_params[prefix + "sigma"].value
            # fraction = fitted_params[prefix + "fraction"].value
            # amp = fitted_params[f"{prefix}amp_t{t_example}"].value
            amp = fitted_params[f"amp_t{t_example}"].value
            
            # Use lmfit model for correct evaluation
            model = LorentzianModel(prefix=prefix)
            temp_params = Parameters()
            temp_params.add(prefix + 'amplitude', value=amp)
            temp_params.add(prefix + 'center', value=center)
            temp_params.add(prefix + 'sigma', value=sigma)
            # temp_params.add(prefix + 'fraction', value=fraction)
            
            y_comp = model.eval(temp_params, x=x_sub)
            line, = ax_main.plot(x_sub, y_comp, '--', lw=1.5, alpha=0.7,
                                label=f'Peak {i+1} (trace {t_example})')
            fig.component_lines.append(line)
        
        # Baseline for first trace
        # y_baseline = fitted_params[f"bkg_t{t_example}"].value * np.ones_like(x_sub)
        # line, = ax_main.plot(x_sub, y_baseline, ':', lw=1.5, color='k',
        #                     label=f'Baseline (trace {t_example})')
        # fig.component_lines.append(line)
        
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
                # "fraction": fitted_params[prefix + "fraction"].value,
            })
        
        amplitudes_by_trace = []
        # baselines_by_trace = []
        
        for t in range(n_traces):
            # amps = [fitted_params[f"p{i}_amp_t{t}"].value for i in range(n_peaks)]
            amps = [fitted_params[f"amp_t{t}"].value for i in range(n_peaks)]
            amplitudes_by_trace.append(amps)
            # baselines_by_trace.append(fitted_params[f"bkg_t{t}"].value)
        
        window_state["fit_results"] = {
            "x": x_sub,
            "y_fit": y_fit,
            "component_params": component_params,
            "amplitudes_by_trace": amplitudes_by_trace,
            # "baselines_by_trace": baselines_by_trace,
            "areas": areas,
            "chisqr": result.chisqr,
            "redchi": result.redchi,
            "n_peaks": n_peaks,
            "n_traces": n_traces
        }
    
    btn.on_clicked(on_button)
    
    def on_close(event):
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight", dpi=300)
            print(f"\nFigure saved to {savepath}")
    
    fig.canvas.mpl_connect("close_event", on_close)
    
    plt.show()
    
    return window_state