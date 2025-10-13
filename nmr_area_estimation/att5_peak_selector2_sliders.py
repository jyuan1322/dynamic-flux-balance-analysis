import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from lmfit.models import LorentzianModel, ConstantModel
from scipy.signal import find_peaks

def interactive_peak_selector(x_data, y_data, ref_ppm, label,
                             prominence_factor=0.05, base_fit_window=0.04,
                             init_bounds=None, seed=101, savepath=None):
    """
    Interactive viewer with sliders for defining a region and a button
    to refit a composite Lorentzian model inside that region.
    """

    # --- ensure ascending x for fitting ---
    if x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data = y_data[::-1]

    # --- initial plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.3)  # space for widgets

    ax.plot(x_data, y_data, "b.", label="Data")
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()
    ax.legend()

    # --- sliders for region selection ---
    axcolor = "lightgoldenrodyellow"
    ax_left = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_right = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)

    window_state = {"seed": seed}
    if init_bounds is None:
        window_state["lower_ppm_bound"] = np.min(x_data)
        window_state["upper_ppm_bound"] = np.max(x_data)
    else:
        window_state["lower_ppm_bound"],  window_state["upper_ppm_bound"] = init_bounds
    s_left = Slider(ax_left, "Lower PPM bound (Right)", np.min(x_data), np.max(x_data),
                    valinit=window_state["lower_ppm_bound"])
    s_right = Slider(ax_right, "Upper PPM bound (Left)", np.min(x_data), np.max(x_data),
                     valinit=window_state["upper_ppm_bound"])

    vline_left = ax.axvline(s_left.val, color="g", linestyle="--")
    vline_right = ax.axvline(s_right.val, color="g", linestyle="--")

    fit_line, = ax.plot([], [], "r-", lw=2, label="Refit")

    def update_lines(val=None):
        left, right = sorted([s_left.val, s_right.val])
        vline_left.set_xdata([left, left])
        vline_right.set_xdata([right, right])
        fig.canvas.draw_idle()

    s_left.on_changed(update_lines)
    s_right.on_changed(update_lines)

    # --- button for refitting ---
    ax_button = plt.axes([0.4, 0.02, 0.2, 0.05])
    btn = Button(ax_button, "Refit region")

    # --- TextBox for number of Lorentzians ---
    ax_text = plt.axes([0.15, 0.025, 0.15, 0.06])  # [left, bottom, width, height]
    text_box = TextBox(ax_text, "n_peaks", initial="")

    def on_button(event):
        window_state["seed"] += 1
        np.random.seed(window_state["seed"])
        print(f"Random seed set to {window_state['seed']}")

        left, right = sorted([s_left.val, s_right.val]) # inner boundary (sliders)
        outer_left, outer_right = left - 0.05, right + 0.05  # expand for fitting
        window_state["lower_ppm_bound"] = left
        window_state["upper_ppm_bound"] = right
        # mask = (x_data >= left) & (x_data <= right)
        mask = (x_data >= outer_left) & (x_data <= outer_right)
        if np.sum(mask) < 5:
            print("Too few points in region")
            return
        x_sub, y_sub = x_data[mask], y_data[mask]

        # --- peak detection inside region ---
        peaks_idx, _ = find_peaks(
            y_sub, prominence=prominence_factor * (y_sub.max() - y_sub.min())
        )
        if len(peaks_idx) == 0:
            peaks_idx = [np.argmax(y_sub)]

        max_num_peaks = 10
        if len(peaks_idx) > max_num_peaks:
            distances = np.abs(x_sub[peaks_idx] - ref_ppm)
            closest_idx = np.argsort(distances)[:max_num_peaks]
            peaks_idx = [peaks_idx[i] for i in closest_idx]

        # --- Get n_peaks from the TextBox or fallback ---
        try:
            n_peaks = int(text_box.text)
        except ValueError:
            print("Invalid input, falling back to auto-detected n_peaks.")
            n_peaks = len(peaks_idx)
            # --- Update the TextBox with auto-detected value ---
            text_box.set_val(str(n_peaks))   # refreshes visible value
        if n_peaks == 1:
            n_peaks += 1

        # --- build composite model ---
        composite_model = None
        for i in range(n_peaks):
            prefix = f"p{i}_"
            model = LorentzianModel(prefix=prefix)
            composite_model = model if composite_model is None else composite_model + model

        baseline = ConstantModel(prefix="bkg_")
        composite_model += baseline
        params = composite_model.make_params()

        # --- initial guesses ---
        print(f"n_peaks = {n_peaks}, detected peaks = {len(peaks_idx)}")
        for i in range(n_peaks):
            prefix = f"p{i}_"
            # within the selected region, pick peaks if available, otherwise random
            if i < len(peaks_idx):
                peak_idx = peaks_idx[i]
                center_guess = x_sub[peak_idx] + np.random.uniform(-0.01, 0.01)
                amp_guess = (y_sub[peak_idx] - y_sub.min()) * np.pi * 0.005
                amp_guess *= np.random.uniform(0.8, 1.2)
            else:
                center_guess = ref_ppm + np.random.uniform(-0.03, 0.03)
                amp_guess = (y_sub.max() - y_sub.min()) * np.pi * 0.005
                amp_guess *= np.random.uniform(0.5, 1.0)

            params[prefix + "center"].set(value=center_guess,
                                          min=x_sub.min(), max=x_sub.max())
            params[prefix + "amplitude"].set(value=amp_guess, min=0)

            sigma_guess = 0.005 * np.random.uniform(0.8, 1.2)
            params[prefix + "sigma"].set(value=sigma_guess, min=0.001, max=0.03)

        params["bkg_c"].set(value=y_sub.min(),
                            min=y_sub.min()*0.5, max=y_sub.max()*0.5)

        # --- fit ---
        result = composite_model.fit(y_sub, params, x=x_sub)

        # --- save fitted components and parameters ---
        components = []
        component_params = []

        for i in range(n_peaks):
            prefix = f"p{i}_"
            y_comp = composite_model.components[i].eval(result.params, x=x_sub)
            components.append(y_comp)
            component_params.append({
                "amplitude": result.params[prefix+"amplitude"].value,
                "center": result.params[prefix+"center"].value,
                "sigma": result.params[prefix+"sigma"].value
            })

        baseline_array = np.full_like(x_sub, result.params["bkg_c"].value)
        baseline_params = {"c": result.params["bkg_c"].value}

        window_state["fit_results"] = {
            "x": x_sub,
            "best_fit": result.best_fit.copy(),
            "components": components,
            "baseline": baseline_array,
            "component_params": component_params,
            "baseline_params": baseline_params
        }

        # --- plot best fit ---
        fit_line.set_data(x_sub, result.best_fit)

        # --- plot individual components ---
        # Remove old component lines if they exist
        if hasattr(fig, 'component_lines'):
            for line in fig.component_lines:
                line.remove()
        fig.component_lines = []

        # Plot each Lorentzian component
        for i in range(n_peaks):
            prefix = f'p{i}_'
            y_comp = composite_model.components[i].eval(result.params, x=x_sub)
            line, = ax.plot(x_sub, y_comp, linestyle='--', lw=1.5, label=f'Peak {i+1}')
            fig.component_lines.append(line)

        # Plot constant baseline
        y_baseline = result.params['bkg_c'].value * np.ones_like(x_sub)
        line, = ax.plot(x_sub, y_baseline, linestyle=':', lw=1.5, color='k', label='Baseline')
        fig.component_lines.append(line)

        # Redraw axes
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

        print(f"Refit inside region [{left:.3f}, {right:.3f}] ppm")
        for i, params_dict in enumerate(component_params):
            print(f"Peak {i+1}: {params_dict}")
        print(f"Baseline: {baseline_params}")

        # calculate total area of the fit
        # fit_results = window_state["fit_results"]
        # x = fit_results["x"]
        # y = fit_results["best_fit"]
        # total_area = np.trapz(y, x)
        # window_state["total_area"] = total_area
        # calculate total area, only from peaks inside INNER region
        total_area = 0.0
        inner_mask = (x_sub >= left) & (x_sub <= right)
        for i, comp in enumerate(components):
            peak_center = component_params[i]["center"]
            if left <= peak_center <= right:  # only keep peaks whose apex is inside inner window
                total_area += np.trapz(comp[inner_mask], x_sub[inner_mask])
        window_state["total_area"] = total_area
        print(f"Total area under fitted curve: {total_area}")


        # --- build composite curve of included peaks only ---
        inner_left, inner_right = left, right
        inner_mask = (x_sub >= inner_left) & (x_sub <= inner_right)

        included_components = []
        for i, comp in enumerate(components):
            peak_center = component_params[i]["center"]
            if inner_left <= peak_center <= inner_right:
                included_components.append(comp)

        if included_components:
            included_sum = np.sum(included_components, axis=0)
        else:
            included_sum = np.zeros_like(x_sub)

        # plot included composite (remove old one if exists)
        if hasattr(fig, "included_line"):
            fig.included_line.remove()
        fig.included_line, = ax.plot(
            x_sub, included_sum, "m-", lw=2, label="Included peaks"
        )

    btn.on_clicked(on_button)

    # Save when the window is closed
    def on_close(event):
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight")
            print(f"Figure saved to {savepath}")

    fig.canvas.mpl_connect("close_event", on_close)

    plt.show()

    return window_state