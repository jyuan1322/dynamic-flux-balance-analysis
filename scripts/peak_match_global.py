import sys, os, json
import pandas as pd
import numpy as np
import configparser
import matplotlib
if sys.platform == "darwin":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from att5_peak_selector_global import (interactive_peak_selector_global,
                                       interactive_background_fitter)
from spectral_alignment import edit_ppm_shifts, apply_ppm_shifts, edit_baseline

# read from config file
config = configparser.ConfigParser()
config.optionxform = str
# config.read("config/config_UGA_HRMAS_13C_Cells.ini")
config.read("config/config_UGA_HRMAS_13C_Cells_1H_standard.ini")


plt.close('all')

working_dir       = config.get("paths", "working_dir")
input_stack       = os.path.join(working_dir, config.get("paths", "input_stack"))
input_ref_peaks   = os.path.join(working_dir, config.get("paths", "input_ref_peaks"))
output_dir        = config.get("paths", "output_dir")
os.makedirs(output_dir, exist_ok=True)

base_fit_window   = config.getfloat("params", "base_fit_window")
prominence_factor = config.getfloat("params", "prominence_factor")

# Parse trace masking
keep_indices = None
if config.has_option("params", "keep_trace_indices"):
    keep_indices_str = config.get("params", "keep_trace_indices")
    if keep_indices_str.strip():
        keep_indices = []
        for part in keep_indices_str.split(','):
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                keep_indices.extend(range(start, end + 1))
            else:
                keep_indices.append(int(part))
        print(f"\nTrace masking enabled: keeping indices {keep_indices}")

# Load data
df   = pd.read_excel(input_stack, header=None)
data = df.iloc[2:].reset_index(drop=True)
data.columns = ['ppm'] + [f'trace_{i}' for i in range(1, df.shape[1])]
data = data.astype(float)

ppm      = data['ppm'].values
traces   = data.drop(columns='ppm').values
n_traces = traces.shape[1]

if keep_indices is not None:
    trace_mask = np.zeros(n_traces, dtype=bool)
    for i in keep_indices:
        if 0 <= i < n_traces:
            trace_mask[i] = True
    print(f"Trace mask: {np.sum(trace_mask)} active out of {n_traces} total")
else:
    trace_mask = None
    print(f"No trace masking: all {n_traces} traces active")

try:
    real_times = df.iloc[1, 1:df.shape[1]].values.astype(float)
except Exception:
    real_times = None
    print("No real times found.")

ref_peaks = pd.read_csv(input_ref_peaks, sep="\t", header=None,
                        names=["ppm", "label"], comment="#")


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def plot_traces_colorbar(data, ref_ppm, plot_title,
                         base_fit_window=0.04, trace_mask=None):
    ppm_vals = data['ppm'].values
    tr       = data.drop(columns='ppm').values
    n_t      = tr.shape[1]
    fig, ax  = plt.subplots(figsize=(10, 6))
    nidxs    = 20

    if trace_mask is not None:
        active_idx = np.where(trace_mask)[0]
        indices = (active_idx if len(active_idx) <= nidxs
                   else active_idx[np.linspace(0, len(active_idx)-1, nidxs, dtype=int)])
    else:
        indices = np.linspace(0, n_t-1, nidxs, dtype=int)

    colormap = cm.viridis
    norm     = mcolors.Normalize(vmin=0, vmax=n_t-1)
    for t in indices:
        mask = ((ppm_vals >= ref_ppm - base_fit_window) &
                (ppm_vals <= ref_ppm + base_fit_window))
        alpha = 0.8 if (trace_mask is None or trace_mask[t]) else 0.3
        ax.plot(ppm_vals[mask], tr[mask, t], color=colormap(norm(t)), alpha=alpha)

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Trace index")
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.set_title(plot_title)
    ax.invert_xaxis()
    plt.show(block=False)


def calculate_area_global(data, label, ref_ppm, area_scaling_factor=1,
                          real_times=None, exp_name="",
                          base_fit_window=0.04, prominence_factor=0.1,
                          init_bounds=None, seed=101, trace_mask=None):
    """
    Full processing pipeline for one metabolite peak:

      1. edit_ppm_shifts        — interactive per-trace alignment
      2. apply_ppm_shifts       — apply shifts
      3. interactive_background_fitter — Window 1: fit (and optionally skip)
                                         background peaks on a user-chosen region
      4. edit_baseline          — interactive baseline estimation on the
                                  background-subtracted data
      5. interactive_peak_selector_global — Window 2: fit signal peaks on
                                            background-subtracted data
    """
    np.random.seed(seed)

    ppm_axis   = data['ppm'].values
    tr         = data.drop(columns='ppm').values

    # Extract window
    mask       = ((ppm_axis >= ref_ppm - base_fit_window) &
                  (ppm_axis <= ref_ppm + base_fit_window))
    x_data     = ppm_axis[mask]
    y_data_all = tr[mask, :]

    if x_data[0] > x_data[-1]:
        x_data     = x_data[::-1]
        y_data_all = y_data_all[::-1, :]

    nmr_fit_outfile = os.path.join(
        output_dir, f"nmr_fit_global_{exp_name}_{label}_{ref_ppm}.pdf"
    )

    # ---- Step 1 & 2: alignment --------------------------------------------
    ppm_shifts     = edit_ppm_shifts(x_data, y_data_all,
                                     label="Dataset ppm alignment")
    y_aligned      = apply_ppm_shifts(x_data, y_data_all, ppm_shifts)

    # ---- Step 3: background fitting (Window 1) ----------------------------
    # Use the full fitting window as the default region; user can move sliders.
    # Baseline is estimated after background subtraction, so run this first.
    y_background, bkg_state = interactive_background_fitter(
        x_data, y_aligned,
        label=f"{label} {ref_ppm} ppm — Background Fit",
        prominence_factor=prominence_factor,
        init_bounds=init_bounds,
        seed=seed,
        real_times=real_times,
        trace_mask=trace_mask,
        baseline=None,   # no fixed baseline yet — background fitter fits freely
    )
    # y_background is None if user clicked Skip or set n_bkg_peaks=0

    # ---- Step 4: baseline -------------------------------------------------
    # Estimate baseline from background-subtracted data so it reflects the
    # true signal floor rather than the raw (pre-subtraction) floor.
    if y_background is not None:
        y_for_bl_raw = y_aligned - y_background
    else:
        y_for_bl_raw = y_aligned

    if trace_mask is not None:
        y_for_bl = y_for_bl_raw[:, np.where(trace_mask)[0]]
    else:
        y_for_bl = y_for_bl_raw

    baseline_value = edit_baseline(
        x_data, y_for_bl,
        label=f"Baseline Editor — {label} {ref_ppm} ppm"
    )

    # ---- Step 5: signal fitting (Window 2) --------------------------------
    window_state = interactive_peak_selector_global(
        x_data, y_aligned,
        ref_ppm=ref_ppm,
        label=f"{label}, {ref_ppm} ppm — Signal Fit",
        init_bounds=init_bounds,
        seed=seed,
        prominence_factor=prominence_factor,
        base_fit_window=base_fit_window,
        area_scaling_factor=area_scaling_factor,
        savepath=nmr_fit_outfile,
        real_times=real_times,
        trace_mask=trace_mask,
        baseline=baseline_value,
        y_background=y_background,     # None → no subtraction
    )

    # ---- metadata ---------------------------------------------------------
    window_state["experiment_name"] = exp_name
    window_state["reference_peak"]  = float(ref_ppm)
    window_state["metabolite"]      = label
    window_state["scan_depth"]      = len(ppm_axis)
    window_state["n_traces"]        = tr.shape[1]
    window_state["baseline_value"]  = float(baseline_value)
    window_state["background_fit"]  = make_json_serializable(
        {k: v for k, v in bkg_state.items() if k != "fitted_params"}
    )

    if real_times is not None:
        window_state["real_times"] = real_times.tolist()
    if trace_mask is not None:
        window_state["trace_mask_applied"] = trace_mask.tolist()
        window_state["n_active_traces"]    = int(np.sum(trace_mask))

    ws_serial = make_json_serializable(window_state)
    json_outfile = os.path.join(
        output_dir, f"nmr_fit_global_{exp_name}_{label}_{ref_ppm}.json"
    )
    with open(json_outfile, "w") as f:
        json.dump(ws_serial, f, indent=2)
    print(f"\nResults saved to {json_outfile}")
    return window_state


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
init_bounds = None
exp_name    = os.path.splitext(os.path.basename(input_stack))[0]

for _, ref in ref_peaks.iterrows():
    ref_ppm = ref['ppm']
    label   = ref['label']
    print(f"\n{'='*80}")
    print(f"Processing: {label} at {ref_ppm} ppm")
    print(f"{'='*80}")

    init_bounds  = (ref_ppm - base_fit_window / 2,
                    ref_ppm + base_fit_window / 2)
    json_outfile = os.path.join(
        output_dir, f"nmr_fit_global_{exp_name}_{label}_{ref_ppm}.json"
    )

    if os.path.exists(json_outfile):
        print(f"Skipping {label} at {ref_ppm} ppm — already processed.")
        with open(json_outfile, "r") as f:
            window_state = json.load(f)
        init_bounds = (window_state["lower_ppm_bound"],
                       window_state["upper_ppm_bound"])
        continue

    plot_traces_colorbar(data, ref_ppm,
                         plot_title=f"{label} {ref_ppm}",
                         base_fit_window=base_fit_window,
                         trace_mask=trace_mask)

    calculate_area_global(
        data=data, label=label, ref_ppm=ref_ppm,
        area_scaling_factor=1, real_times=real_times,
        exp_name=exp_name, base_fit_window=base_fit_window,
        prominence_factor=prominence_factor,
        init_bounds=init_bounds, seed=101, trace_mask=trace_mask
    )

print("\n" + "="*80)
print("ALL PROCESSING COMPLETE")
print("="*80)