import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, PseudoVoigtModel
import numpy as np

def interactive_peak_selector(x_data, y_data, result, model_type="lorentzian", ref_ppm=None, plot_label=None, savepath=None):
    """
    Interactive viewer to select/deselect individual fitted peaks.

    Parameters
    ----------
    x_data : np.ndarray
        X-axis values (ppm).
    y_data : np.ndarray
        Spectrum intensities in this window.
    result : lmfit.ModelResult
        Fitted model result from lmfit.
    model_type : str
        Type of peak to reconstruct: 'lorentzian', 'gaussian', 'voigt', 'pseudo_voigt'.
    ref_ppm : float, optional
        Reference ppm (for plot title).

    Returns
    -------
    selected_peaks : list[int]
        Indices of peaks the user selected.
    """

    # Choose model factory based on type
    model_map = {
        "lorentzian": LorentzianModel,
        "gaussian": GaussianModel,
        "voigt": VoigtModel,
        "pseudo_voigt": PseudoVoigtModel
    }
    model_class = model_map[model_type.lower()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_data, y_data, "b.", label="Data")
    ax.plot(x_data, result.best_fit, "r-", label="Best fit")

    # Store line objects + selection state
    line_objects = []
    selection_state = {}

    # Collect all peak prefixes from result.params
    peak_prefixes = sorted({p.split("_")[0]+"_" for p in result.params if "center" in p})

    for i, prefix in enumerate(peak_prefixes):
        comp = model_class(prefix=prefix).eval(result.params, x=x_data)
        line, = ax.plot(x_data, comp, "--", label=f"Peak {i+1}",
                        picker=True, pickradius=5)
        line_objects.append(line)
        selection_state[line] = False  # default = included

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.invert_xaxis()
    if ref_ppm is not None:
        ax.set_title(f"{plot_label} peaks near {ref_ppm:.3f} ppm")

    # Toggle function
    def on_pick(event):
        line = event.artist
        if line in selection_state:
            selection_state[line] = not selection_state[line]
            if selection_state[line]:
                line.set_linewidth(2.5)
                line.set_alpha(1.0)
                line.set_color("m")   # highlight
            else:
                line.set_linewidth(1.0)
                line.set_alpha(0.3)
                line.set_color("gray")
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("pick_event", on_pick)

    # Save when the window is closed
    def on_close(event):
        if savepath is not None:
            fig.savefig(savepath, bbox_inches="tight")
            print(f"Figure saved to {savepath}")

    fig.canvas.mpl_connect("close_event", on_close)

    plt.show()

    # Return list of selected peak indices
    selected_peaks = [
        i for i, line in enumerate(line_objects) if selection_state[line]
    ]
    return selected_peaks
