import os, json
import pandas as pd
import numpy as np
import pickle
import configparser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from lmfit import Model
from lmfit.models import LorentzianModel, VoigtModel, ConstantModel
from scipy.signal import find_peaks
from att5_peak_selector2_sliders import interactive_peak_selector

# read from config file
config = configparser.ConfigParser()
config.optionxform = str   # <-- turn off lowercasing
config.read("config.ini")

# Close any existing plots (this also helps recover from Ctrl-C in previous run)
plt.close('all')

### Proline
# working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data1_13CPro1"
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data1_13CPro1"
# input_stack = os.path.join(working_dir, "Data1_13CPro1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data2_13CPro2"
# input_stack = os.path.join(working_dir, "Data2_13CPro2_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data3_13CPro3"
# input_stack = os.path.join(working_dir, "Data3_13CPro3_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")

### 13C 13C_Glucose
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data7_13CGlc1"
# input_stack = os.path.join(working_dir, "Data7_13CGlc1_13C.xlsx")
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data8_13CGlc2"
# input_stack = os.path.join(working_dir, "Data8_13CGlc2_13C.xlsx")
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data9_13CGlc3"
# input_stack = os.path.join(working_dir, "Data9_13CGlc3_13C.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_13C_temp.txt")

### Leucine 1 (deconvolve) 9/21/2025
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data4_13CLeu1"
# input_stack = os.path.join(working_dir, "Data4_13CLeu1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")

### 1H Glc experiments (11/3/2025)
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data7_13CGlc1"
# input_stack = os.path.join(working_dir, "Data7_13CGlc1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")

### 13C Glc product standards
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/20220325_13CGlc_Standards"
# input_stack = os.path.join(working_dir, "traces_13C_annot.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_13C_temp.txt")

### 1H Glc product standards
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/20220325_13CGlc_Standards"
# input_stack = os.path.join(working_dir, "traces_1H_annot.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")

# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data8_13CGlc2"
# input_stack = os.path.join(working_dir, "Data8_13CGlc2_1H.xlsx")
# input_stack = os.path.join(working_dir, "Data8_13CGlc2_13C.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")
# input_ref_peaks = os.path.join(working_dir, "cfg_13C_temp.txt")
# out_csv = os.path.join(working_dir, "peak_areas_glc2_lmfit.csv")

# working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data4_13CLeu1"
# input_stack = os.path.join(working_dir, "Data4_13CLeu1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
# out_csv = os.path.join(working_dir, "peak_areas_leu1_lmfit.csv")

# UGA HRMAS 10/31/2025 1H
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/UGA_HRMAS_10312025"
# input_stack = os.path.join(working_dir, "spectra_1H.pkl")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")
# UGA HRMAS 11/03/2025 1H
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/UGA_HRMAS_11032025"
# input_stack = os.path.join(working_dir, "spectra_1H.pkl")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp_V2.txt")
# UGA HRMAS 11/03/2025 13C
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/UGA_HRMAS_11032025"
# input_stack = os.path.join(working_dir, "spectra_13C.pkl")
# input_ref_peaks = os.path.join(working_dir, "cfg_13C_temp.txt")

working_dir = config.get("paths", "working_dir")
input_stack = os.path.join(working_dir, config.get("paths", "input_stack"))
input_ref_peaks = os.path.join(working_dir, config.get("paths", "input_ref_peaks"))
output_dir = config.get("paths", "output_dir")
os.makedirs(output_dir, exist_ok=True)

# for 1H
# base_fit_window = 0.08
# base_fit_window = 0.04
# base_fit_window= 0.04
base_fit_window = config.getfloat("params", "base_fit_window")
# for 13C
# base_fit_window =0.4
# base_fit_window = 20
# prominence_factor = 0.1
prominence_factor = config.getfloat("params", "prominence_factor")

# Load data
# def run_fit(working_dir, input_stack, input_ref_peaks, out_csv):
df = pd.read_excel(input_stack, header=None)
data = df.iloc[2:].reset_index(drop=True)
data.columns = ['ppm'] + [f'trace_{i}' for i in range(1, df.shape[1])]
data = data.astype(float)

# with open(input_stack, "rb") as f:
#     spectra_dict = pickle.load(f)

# spectra_dict = {int(k.split("_")[0]): v for k, v in spectra_dict.items()}
# # UGA HRMAS 10/31/2025 1H
# # spectra_dict = {k: v for k, v in spectra_dict.items() if k >= 101}
# # spectra_dict = {k: v for k, v in spectra_dict.items() if (k-101) % 5 == 0}  # every 5th time point starting from 31
# UGA HRMAS 11/03/2025 1H
# spectra_dict = {k: v for k, v in spectra_dict.items() if k >= 31}
# spectra_dict = {k: v for k, v in spectra_dict.items() if (k-31) % 5 == 0}  # every 5th time point starting from 31
# # UGA HRMAS 11/03/2025 13C
# # spectra_dict = {k: v for k, v in spectra_dict.items() if k >= 33 and k <= 233}
# # spectra_dict = {k: v for k, v in spectra_dict.items() if (k-33) % 5 == 0}  # every 5th time point starting from 31

ppm = data['ppm'].values
traces = data.drop(columns='ppm').values
n_traces = traces.shape[1]
try:
    real_times = df.iloc[1, 1:df.shape[0]].values.astype(float)
except:
    real_times = None
    print("No real times found, using trace indices as time.")
# except:
#     # standard solution, 102_13C
#     real_times = df.iloc[1, 1:df.shape[0]].values
#     real_times = np.array([s.split("_")[0] for s in real_times]).astype(float)

ref_peaks = pd.read_csv(input_ref_peaks, sep="\t", header=None, names=["ppm", "label"], comment="#")

def make_json_serializable(obj):
    """
    Recursively convert numpy arrays in a dict/list to Python lists.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# def lorentzian_area_lmfit(amplitude, sigma):
#     # lmfit Lorentzian 'sigma' is HWHM (half-width at half max)
#     return np.pi * amplitude * sigma


"""def plot_traces(spectra_dict, ref_ppm, plot_title, base_fit_window=0.04):
    # Sort spectra by integer keys
    spectra_items = sorted(spectra_dict.items(), key=lambda item: item[0])
    n_traces = len(spectra_items)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Continuous color gradient over all traces
    colormap = cm.viridis
    colors = [colormap(i / (n_traces - 1)) for i in range(n_traces)]

    for i, (name, spec) in enumerate(spectra_items):
        ppm = spec["ppm"]
        intensity = spec["intensity"]

        # Mask around the reference peak
        mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
        x_data = ppm[mask]
        y_data = intensity[mask]

        ax.plot(x_data, y_data, color=colors[i], label=f"{name}")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.set_title(plot_title)
    ax.invert_xaxis()
    fig.savefig(f"{plot_title}.pdf", dpi=300, bbox_inches="tight")
    plt.show(block=False)
"""

def plot_traces_colorbar(data, ref_ppm, plot_title, base_fit_window=0.04):
    ppm = data['ppm'].values
    traces = data.drop(columns='ppm').values
    n_traces = traces.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    nidxs = 20  # number of evenly spaced traces
    indices = np.linspace(0, n_traces-1, nidxs, dtype=int)

    # Choose a colormap
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
    sm.set_array([])  # needed for matplotlib < 3.6
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Trace index")

    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.set_title(plot_title)
    ax.invert_xaxis()
    plt.show(block=False)


def calculate_area(data, label, ref_ppm, t, area_scaling_factor=1, real_times=None, exp_name="",
                   base_fit_window=0.04, prominence_factor=0.1, init_bounds=None, seed=101):
    np.random.seed(seed)

    ppm = data['ppm'].values
    traces = data.drop(columns='ppm').values
    # ppm = np.array(data["ppm"])
    # y_data_full = np.array(data["intensity"])

    y = traces[:, t]
    mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
    x_data = ppm[mask]
    y_data = y[mask]
    # y_data = y_data_full[mask]

    # Ensure ascending ppm for lmfit
    if x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data = y_data[::-1]

    nmr_fit_outfile = os.path.join(output_dir, f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.pdf")
    # selected_peaks = interactive_peak_selector(x_data, y_data, comp_model,
    #                                         model_type="lorentzian",
    #                                         ref_ppm=ref_ppm,
    #                                         plot_label=label,
    #                                         savepath=nmr_fit_outfile)
    window_state = interactive_peak_selector(x_data, y_data,
                                ref_ppm=ref_ppm, label=f"{label}, {ref_ppm}, fid {t}",
                                init_bounds=init_bounds, seed=seed,
                                prominence_factor=prominence_factor,
                                base_fit_window=base_fit_window,
                                area_scaling_factor=area_scaling_factor,
                                savepath=nmr_fit_outfile)

    # add the trace index and experiment name to the saved state
    window_state["trace_index"] = int(t)
    # if real_times is not None:
    window_state["time"] = float(real_times[t])
    # else:
    #     window_state["time"] = None
    window_state["experiment_name"] = exp_name
    window_state["reference_peak"] = float(ref_ppm)
    window_state["metabolite"] = label
    window_state["scan_depth"] = len(ppm)

    # make this json serializable
    window_state_serializable = make_json_serializable(window_state)
    print(window_state_serializable)

    json_outfile = os.path.join(output_dir, f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.json")
    with open(json_outfile, "w") as f:
        json.dump(window_state_serializable, f, indent=2)
    return window_state


# results = []
# base_fit_window = 0.04  # window half-width around reference peak
init_bounds = None  # (lower_ppm, upper_ppm) or None for full range
for _, ref in ref_peaks.iterrows():
    ref_ppm = ref['ppm']
    label = ref['label']
    print(f"{label} {ref_ppm}")

    plot_traces_colorbar(data, ref_ppm, plot_title=f"{label} {ref_ppm}", base_fit_window = base_fit_window)
    # plot_traces_colorbar(data, ref_ppm, real_times, plot_title=f"{label} {ref_ppm}", base_fit_window = base_fit_window)
    
    # plot_traces(data, ref_ppm, real_times, plot_title=f"{label} {ref_ppm}", base_fit_window = base_fit_window)
    # plot_traces(spectra_dict, ref_ppm, plot_title=f"{label}_{ref_ppm}", base_fit_window = base_fit_window)


    # This scaling factor was to accommodate for different acquisition depths, which linearly scales the area 
    # base_scaling_factor = None
    area_scaling_factor = 1
    # for i, (sample_name, spec) in enumerate(spectra_dict.items()):
    for t in range(n_traces):
        # if base_scaling_factor is None:
        #     base_scaling_factor = len(spec["ppm"])
        # else:
        #     area_scaling_factor = len(spec["ppm"]) / base_scaling_factor
        exp_name = os.path.splitext(os.path.basename(input_stack))[0]
        json_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.json"
        # json_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{sample_name}.json"
        if os.path.exists(json_outfile):
            print(f"Skipping trace {t}/{n_traces} for peak {label} at {ref_ppm} ppm in {exp_name}, already done.")
            # print(f"Skipping {sample_name} ({i+1}/{len(spectra_dict)}) for peak {label} at {ref_ppm} ppm — already done.")
            with open(json_outfile, "r") as f:
                window_state = json.load(f)
            # results.extend(results_sub)
            init_bounds = (window_state["lower_ppm_bound"], window_state["upper_ppm_bound"])
            continue
        else:
            print("-" * 80)
            print(f"Fitting trace {t}/{n_traces} for peak {label} at {ref_ppm} ppm in {exp_name}")
            # print(f"Fitting {sample_name} ({i+1}/{len(spectra_dict)}) for peak {label} at {ref_ppm} ppm")
            # ppm = spec["ppm"]
            # intensity = spec["intensity"]
            # window_state = calculate_area(data={"ppm": ppm, "intensity": intensity}, label=label, ref_ppm=ref_ppm,
            window_state = calculate_area(data=data, label=label, ref_ppm=ref_ppm,
                                          t=t, area_scaling_factor=area_scaling_factor,
                                          real_times=real_times, exp_name = exp_name, base_fit_window=base_fit_window,
                                          prominence_factor=prominence_factor, init_bounds=init_bounds, seed=101)
            init_bounds = (window_state["lower_ppm_bound"], window_state["upper_ppm_bound"])
            # results.extend(results_sub)
            print("-" * 80)

# Save results
# out_df = pd.DataFrame(results)
# out_df.to_csv(out_csv, index=False)
# print(out_df.head())
# return out_df
