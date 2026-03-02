import os, json
import pandas as pd
import numpy as np
import pickle
import configparser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from att5_peak_selector_global import interactive_peak_selector_global
from spectral_alignment import edit_ppm_shifts, apply_ppm_shifts

# read from config file
config = configparser.ConfigParser()
config.optionxform = str   # <-- turn off lowercasing
# config.read("config.ini")
# 13C
# config.read("config_feb052026_UGA_HRMAS_13C_Cells.ini")
# 1H standards (fid 21)
config.read("config_feb052026_UGA_HRMAS_13C_Cells_1H_standard2.ini")
# 1H mixture (fid 25)
# config.read("config_feb052026_UGA_HRMAS_13C_Cells_1H_mixture.ini")

# Close any existing plots
plt.close('all')

working_dir = config.get("paths", "working_dir")
input_stack = os.path.join(working_dir, config.get("paths", "input_stack"))
input_ref_peaks = os.path.join(working_dir, config.get("paths", "input_ref_peaks"))
output_dir = config.get("paths", "output_dir")
os.makedirs(output_dir, exist_ok=True)

base_fit_window = config.getfloat("params", "base_fit_window")
prominence_factor = config.getfloat("params", "prominence_factor")

# Load data
df = pd.read_excel(input_stack, header=None)
data = df.iloc[2:].reset_index(drop=True)
data.columns = ['ppm'] + [f'trace_{i}' for i in range(1, df.shape[1])]
data = data.astype(float)

ppm = data['ppm'].values
traces = data.drop(columns='ppm').values
n_traces = traces.shape[1]

try:
    real_times = df.iloc[1, 1:df.shape[1]].values.astype(float)
except:
    real_times = None
    print("No real times found, using trace indices as time.")

ref_peaks = pd.read_csv(input_ref_peaks, sep="\t", header=None, 
                        names=["ppm", "label"], comment="#")

def make_json_serializable(obj):
    """Recursively convert numpy arrays to lists."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def plot_traces_colorbar(data, ref_ppm, plot_title, base_fit_window=0.04):
    """Plot a subset of traces with colorbar"""
    ppm = data['ppm'].values
    traces = data.drop(columns='ppm').values
    n_traces = traces.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    nidxs = 20  # number of evenly spaced traces
    indices = np.linspace(0, n_traces-1, nidxs, dtype=int)

    colormap = cm.viridis
    norm = mcolors.Normalize(vmin=indices.min(), vmax=indices.max())

    for t in indices:
        y = traces[:, t]
        mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
        x_data = ppm[mask]
        y_data = y[mask]
        ax.plot(x_data, y_data, color=colormap(norm(t)))

    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Trace index")

    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.set_title(plot_title)
    ax.invert_xaxis()
    plt.show(block=False)

def calculate_area_global(data, label, ref_ppm, area_scaling_factor=1, 
                         real_times=None, exp_name="",
                         base_fit_window=0.04, prominence_factor=0.1, 
                         init_bounds=None, seed=101):
    """
    Perform global fitting across all FIDs for a given reference peak.
    """
    np.random.seed(seed)

    ppm = data['ppm'].values
    traces = data.drop(columns='ppm').values
    n_traces = traces.shape[1]

    # Extract region around reference peak
    mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
    x_data = ppm[mask]
    y_data_all = traces[mask, :]

    # Ensure ascending ppm for lmfit
    if x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data_all = y_data_all[::-1, :]

    nmr_fit_outfile = os.path.join(output_dir, 
                                   f"nmr_fit_global_{exp_name}_{label}_{ref_ppm}.pdf")
    
    ppm_shifts = edit_ppm_shifts(
        x_data,
        y_data_all,
        label="Dataset ppm alignment"
    )

    # APPLY SHIFTS HERE
    y_data_aligned = apply_ppm_shifts(
        x_data,
        y_data_all,
        ppm_shifts
    )

    window_state = interactive_peak_selector_global(
        x_data, y_data_aligned,
        ref_ppm=ref_ppm, 
        label=f"{label}, {ref_ppm} ppm - GLOBAL FIT",
        init_bounds=init_bounds, 
        seed=seed,
        prominence_factor=prominence_factor,
        base_fit_window=base_fit_window,
        area_scaling_factor=area_scaling_factor,
        savepath=nmr_fit_outfile,
        real_times=real_times
    )

    # Add metadata to the saved state
    window_state["experiment_name"] = exp_name
    window_state["reference_peak"] = float(ref_ppm)
    window_state["metabolite"] = label
    window_state["scan_depth"] = len(ppm)
    window_state["n_traces"] = n_traces
    
    # Add real times if available
    if real_times is not None:
        window_state["real_times"] = real_times.tolist()
    
    # Make JSON serializable
    window_state_serializable = make_json_serializable(window_state)

    json_outfile = os.path.join(output_dir, 
                                f"nmr_fit_global_{exp_name}_{label}_{ref_ppm}.json")
    with open(json_outfile, "w") as f:
        json.dump(window_state_serializable, f, indent=2)
    
    print(f"\nResults saved to {json_outfile}")
    
    return window_state

# Main loop over reference peaks
init_bounds = None
exp_name = os.path.splitext(os.path.basename(input_stack))[0]

for _, ref in ref_peaks.iterrows():
    ref_ppm = ref['ppm']
    label = ref['label']
    print(f"\n{'='*80}")
    print(f"Processing: {label} at {ref_ppm} ppm")
    print(f"{'='*80}")

    # DEFAULT bounds for this metabolite
    init_bounds = (
        ref_ppm - base_fit_window / 2,
        ref_ppm + base_fit_window / 2
    )

    # Check if already processed
    json_outfile = os.path.join(output_dir,
                                f"nmr_fit_global_{exp_name}_{label}_{ref_ppm}.json")
    
    if os.path.exists(json_outfile):
        print(f"Skipping {label} at {ref_ppm} ppm - already processed.")
        with open(json_outfile, "r") as f:
            window_state = json.load(f)
        init_bounds = (window_state["lower_ppm_bound"], window_state["upper_ppm_bound"])
        continue

    # Plot overview
    plot_traces_colorbar(data, ref_ppm, 
                        plot_title=f"{label} {ref_ppm}", 
                        base_fit_window=base_fit_window)

    # Perform global fitting
    area_scaling_factor = 1
    window_state = calculate_area_global(
        data=data, 
        label=label, 
        ref_ppm=ref_ppm,
        area_scaling_factor=area_scaling_factor,
        real_times=real_times, 
        exp_name=exp_name, 
        base_fit_window=base_fit_window,
        prominence_factor=prominence_factor, 
        init_bounds=init_bounds, 
        seed=101
    )

    # originally, this was meant to carry the bounds to the next time point for the same metabolite
    # init_bounds = (window_state["lower_ppm_bound"], window_state["upper_ppm_bound"])

print("\n" + "="*80)
print("ALL PROCESSING COMPLETE")
print("="*80)