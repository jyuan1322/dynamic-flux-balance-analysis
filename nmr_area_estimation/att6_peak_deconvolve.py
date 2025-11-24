import os, json, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from lmfit import Model
from lmfit.models import LorentzianModel, VoigtModel, ConstantModel
from scipy.signal import find_peaks
from att5_peak_selector2_sliders import interactive_peak_selector
# from att6_deconv_stan_timeseries import run_nmr_model
from att6_deconv2_EMMAP_timeseries2 import auto_init_from_mean
from att6_deconv2_EMMAP_timeseries3 import *

# Close any existing plots (this also helps recover from Ctrl-C in previous run)
plt.close('all')

# working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data1_13CPro1"
# input_stack = os.path.join(working_dir, "Data1_13CPro1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
# out_csv = os.path.join(working_dir, "peak_areas_pro1_lmfit.csv")

# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data7_13CGlc1"
# input_stack = os.path.join(working_dir, "Data7_13CGlc1_13C.xlsx")
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data8_13CGlc2"
# input_stack = os.path.join(working_dir, "Data8_13CGlc2_13C.xlsx")
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data9_13CGlc3"
# input_stack = os.path.join(working_dir, "Data9_13CGlc3_13C.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_13C_temp.txt")

working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data4_13CLeu1"
input_stack = os.path.join(working_dir, "Data4_13CLeu1_1H.xlsx")
input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")

# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data5_13CLeu2"
# input_stack = os.path.join(working_dir, "Data5_13CLeu2_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")

# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data6_13CLeu3"
# input_stack = os.path.join(working_dir, "Data6_13CLeu3_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")

# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data7_13CGlc1"
# input_stack = os.path.join(working_dir, "Data7_13CGlc1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")

# 13C Glc product standards
# working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/20220325_13CGlc_Standards"
# input_stack = os.path.join(working_dir, "traces_13C_annot.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_13C_temp.txt")

# 1H Glc product standards
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

# for 1H
# base_fit_window = 0.08
base_fit_window = 0.08
# base_fit_window = 0.2
# for 13C
# base_fit_window =0.4
# base_fit_window = 100
prominence_factor = 0.1

# Load data
# def run_fit(working_dir, input_stack, input_ref_peaks, out_csv):
df = pd.read_excel(input_stack, header=None)
data = df.iloc[2:].reset_index(drop=True)
data.columns = ['ppm'] + [f'trace_{i}' for i in range(1, df.shape[1])]
data = data.astype(float)

ppm = data['ppm'].values
traces = data.drop(columns='ppm').values
n_traces = traces.shape[1]
try:
    real_times = df.iloc[1, 1:df.shape[0]].values.astype(float)
except:
    # standard solution, 102_13C
    real_times = df.iloc[1, 1:df.shape[0]].values
    real_times = np.array([s.split("_")[0] for s in real_times]).astype(float)

ref_peaks = pd.read_csv(input_ref_peaks, sep="\t", header=None, names=["ppm", "label"])

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

def lorentzian_area_lmfit(amplitude, sigma):
    # lmfit Lorentzian 'sigma' is HWHM (half-width at half max)
    return np.pi * amplitude * sigma


def plot_traces(data, ref_ppm, real_times, plot_title, base_fit_window=0.04):
    ppm = data['ppm'].values
    traces = data.drop(columns='ppm').values
    n_traces = traces.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    nidxs = 20  # number of evenly spaced traces
    indices = np.linspace(0, n_traces-1, nidxs, dtype=int)

    # Choose a colormap
    colormap = cm.viridis  # you can pick 'plasma', 'cividis', 'magma', etc.
    colors = [colormap(i / (nidxs - 1)) for i in range(nidxs)]

    for i, t in enumerate(indices):
        y = traces[:, t]
        mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
        x_data = ppm[mask]
        y_data = y[mask]
        # ax.plot(x_data, y_data, color=colors[i], label=f'Trace {t}')
        ax.plot(x_data, y_data, color=colors[i], label=f'Trace {real_times[t]}')

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.set_title(plot_title)
    ax.invert_xaxis()
    plt.show(block=False)

def plot_traces_colorbar(data, ref_ppm, real_times, plot_title, base_fit_window=0.04):
    ppm = data['ppm'].values
    traces = data.drop(columns='ppm').values
    n_traces = traces.shape[1]

    fig, ax = plt.subplots(figsize=(10, 6))

    nidxs = 20  # number of evenly spaced traces
    indices = np.linspace(0, n_traces-1, nidxs, dtype=int)

    # Choose a colormap
    colormap = cm.viridis
    # norm = mcolors.Normalize(vmin=indices.min(), vmax=indices.max())
    norm = mcolors.Normalize(vmin=real_times.min(), vmax=real_times.max())

    for t in indices:
        y = traces[:, t]
        mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
        x_data = ppm[mask]
        y_data = y[mask]
        ax.plot(x_data, y_data, color=colormap(norm(real_times[t])))

    # Add colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # needed for matplotlib < 3.6
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", label="Trace index")

    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.set_title(plot_title)
    ax.invert_xaxis()
    plt.show(block=False)

def calculate_area(data, label, ref_ppm, t, real_times, exp_name="", base_fit_window=0.04, prominence_factor=0.1,
                   init_bounds=None, seed=101):
    np.random.seed(seed)

    ppm = data['ppm'].values
    traces = data.drop(columns='ppm').values

    y = traces[:, t]
    mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
    x_data = ppm[mask]
    y_data = y[mask]

    # Ensure ascending ppm for lmfit
    if x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data = y_data[::-1]

    nmr_fit_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.pdf"
    # selected_peaks = interactive_peak_selector(x_data, y_data, comp_model,
    #                                         model_type="lorentzian",
    #                                         ref_ppm=ref_ppm,
    #                                         plot_label=label,
    #                                         savepath=nmr_fit_outfile)
    window_state = interactive_peak_selector(x_data, y_data,
                                ref_ppm=ref_ppm, label=label,
                                init_bounds=init_bounds, seed=seed,
                                prominence_factor=prominence_factor,
                                base_fit_window=base_fit_window,
                                savepath=nmr_fit_outfile)

    # add the trace index and experiment name to the saved state
    window_state["trace_index"] = int(t)
    window_state["time"] = float(real_times[t])
    window_state["experiment_name"] = exp_name
    window_state["reference_peak"] = float(ref_ppm)
    window_state["metabolite"] = label

    # make this json serializable
    window_state_serializable = make_json_serializable(window_state)
    print(window_state_serializable)

    json_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.json"
    with open(json_outfile, "w") as f:
        json.dump(window_state_serializable, f, indent=2)
    return window_state


# results = []
# base_fit_window = 0.04  # window half-width around reference peak
init_bounds = None  # (lower_ppm, upper_ppm) or None for full range
# for _, ref in ref_peaks.iterrows():


ref = ref_peaks.iloc[0, :]
ref_ppm = ref['ppm']
label = ref['label']
print(f"{label} {ref_ppm}")

# plot_traces_colorbar(data, ref_ppm, real_times, plot_title=f"{label} {ref_ppm}", base_fit_window = base_fit_window)

ppm = data['ppm'].values
traces = data.drop(columns='ppm').values

# y = traces[:, t]
# mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)

# Normally, you would refine the mask using the interface
# For now, just hard-code the refined mask
# Leu1
mask = (ppm >= 0.82) & (ppm <= 0.87)
x_data = ppm[mask]
y_data = traces[mask, :] # ppm x time point

seed=104 # 104 n_peaks = 6, 102 n_peaks=6, 102 n_peaks=10
n_peaks = 6
np.random.seed(seed)
random.seed(seed)

# initialize (from mean)
centers0, gammas0 = auto_init_from_mean(x=x_data, Y=y_data, n_peaks=n_peaks, gamma_guess=0.005)
centers0 += np.random.normal(0, 0.01, size=centers0.shape)  # jitter
gammas0 *= 1.1

# test EM MAP solution across all time points
"""
# lambda_a = 1e-4
centers, gammas, amps_est, baselines_est, history = (
    em_map_deconvolution(x = x_data,
                        Y = y_data,
                        K_init = None,
                        centers_init = centers0,
                        gammas_init = gammas0,
                        n_iter=20,
                        lambda_a=1e-3, a_prior=None,
                        lambda_mu=0.0, mu_prior=None,
                        lambda_gamma=1e-3, gamma_prior=0.01,
                        nonneg=True,
                        verbose=True)
)
"""
# x, Y defined earlier
centers, gammas, pis, Ss, bs, history = em_softmax_mixture_shapes(
    x = x_data, Y = y_data, centers0 = centers0, gammas0 = gammas0, n_iter=20,
    lambda_pi=1e-5, pi_prior=np.ones(len(centers0))/len(centers0),
    lambda_mu=1e-6, lambda_gamma=1e-6,
    bounds_logS=(-10,10), bounds_b=(-0.05,0.05),
    verbose=True
)


# visualize with wide mask
mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
x = ppm[mask]
Y = traces[mask, :] # ppm x time point
K_final = build_K(x, centers, gammas)
"""
for ti in [0, 36, 10, 20, 30]:
    # reconstruct spectrum
    recon = K_final @ amps_est[:, ti] + baselines_est[ti]
    
    plt.figure(figsize=(8,4))
    plt.plot(x, Y[:, ti], label=f'data t={ti}')
    plt.plot(x, recon, label='recon', lw=2)
    
    # plot each component contribution
    for i in range(K_final.shape[1]):
        plt.plot(x, amps_est[i, ti] * K_final[:, i], '--', label=f'comp{i}')
    
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()
"""
for ti in [0, 36, 10, 20, 30]:
    # compute component amplitudes from softmax mixture
    amps = Ss[ti] * pis[:, ti]   # shape (n_components,)
    
    # full reconstruction
    recon = K_final @ amps + bs[ti]
    
    plt.figure(figsize=(8,4))
    plt.plot(x, Y[:, ti], label=f'data t={ti}')
    plt.plot(x, recon, label='recon', lw=2)
    
    # individual component contributions
    for i in range(K_final.shape[1]):
        plt.plot(x, amps[i] * K_final[:, i], '--', label=f'comp{i}')
    
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()




# print errors across iterations
errors = [h["err"] for h in history]
plt.plot(errors, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.title("Error over iterations")
plt.show()

# print amplitudes across time
"""
times = np.arange(amps_est.shape[1])  # or your actual time values
plt.figure(figsize=(8,5))

for i in range(amps_est.shape[0]):
    plt.plot(times, amps_est[i, :], label=f'Comp {i}')

plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("Component amplitudes across time")
plt.legend()
plt.show()
"""
# Suppose pis has shape (K, n_times)
# and times is an array of timepoints (n_times,)
K, n_times = pis.shape
times = np.arange(n_times)  # replace with your actual time values

plt.figure(figsize=(8, 5))
for k in range(K):
    plt.plot(times, pis[k, :], marker="o", label=f"Peak {k}")

plt.xlabel("Time")
plt.ylabel("Mixture weight (π)")
plt.title("Lorentzian mixture weights over time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----

"""
exp_name = os.path.splitext(os.path.basename(input_stack))[0]
t_final = y_data.shape[1] - 1
window_states = []
for t in [0, t_final]:
    json_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.json"
    if os.path.exists(json_outfile):
        print(f"Skipping trace {t}/{n_traces} for peak {label} at {ref_ppm} ppm in {exp_name}, already done.")
        with open(json_outfile, "r") as f:
            window_state = json.load(f)
        init_bounds = (window_state["lower_ppm_bound"], window_state["upper_ppm_bound"])
    else:
        print("-" * 80)
        print(f"Fitting trace {t}/{n_traces} for peak {label} at {ref_ppm} ppm in {exp_name}")
        window_state = calculate_area(data=data, label=label, ref_ppm=ref_ppm, t=t, real_times=real_times,
                                            exp_name = exp_name, base_fit_window=base_fit_window,
                                            prominence_factor=prominence_factor, init_bounds=init_bounds, seed=101)
        init_bounds = (window_state["lower_ppm_bound"], window_state["upper_ppm_bound"])
    window_states.append(window_state)

# update the boundaries
mask = (ppm >= init_bounds[0]) & (ppm <= init_bounds[1])
x_data = ppm[mask]
y_data = traces[mask, :]

Y = y_data.transpose()
ppm = x_data
first_fit = window_states[0]
last_fit = window_states[-1]
"""





"""
run_nmr_model(Y = y_data.transpose(),
              ppm = x_data,
              first_fit = window_states[0],
              last_fit = window_states[-1],
              num_samples=1000, num_chains=4)
"""

"""

    # Ensure ascending ppm for lmfit
    if x_data[0] > x_data[-1]:
        x_data = x_data[::-1]
        y_data = y_data[::-1]


    for t in range(n_traces):
        exp_name = os.path.splitext(os.path.basename(input_stack))[0]
        json_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.json"
        if os.path.exists(json_outfile):
            print(f"Skipping trace {t}/{n_traces} for peak {label} at {ref_ppm} ppm in {exp_name}, already done.")
            with open(json_outfile, "r") as f:
                window_state = json.load(f)
            # results.extend(results_sub)
            init_bounds = (window_state["lower_ppm_bound"], window_state["upper_ppm_bound"])
            continue
        else:
            print("-" * 80)
            print(f"Fitting trace {t}/{n_traces} for peak {label} at {ref_ppm} ppm in {exp_name}")
            window_state = calculate_area(data=data, label=label, ref_ppm=ref_ppm, t=t, real_times=real_times,
                                        exp_name = exp_name, base_fit_window=base_fit_window,
                                        prominence_factor=prominence_factor, init_bounds=init_bounds, seed=101)
            init_bounds = (window_state["lower_ppm_bound"], window_state["upper_ppm_bound"])
            # results.extend(results_sub)
            print("-" * 80)

# Save results
# out_df = pd.DataFrame(results)
# out_df.to_csv(out_csv, index=False)
# print(out_df.head())
# return out_df
"""




"""
# example window_state
{
  "experiment_name": "Data4_13CLeu1_1H",
  "fit_results": {
    "baseline": "ndarray(120): [1094.565552101591, 1094.565552101591, 1094.565552101591, 1094.565552101591, 1094.565552101591]...",
    "baseline_params": {"c": 1094.565552101591},
    "best_fit": "ndarray(120): [2759.98128731, 2814.5914293, 2871.90743305, 2932.10845322, 2995.38840114]...",
    "component_params": [
      {"amplitude": 595.7105270911644, "center": 0.8512934124166947, "sigma": 0.009879821793953185},
      {"amplitude": 472.0701007794595, "center": 0.8392844307868085, "sigma": 0.010532633768274093}
    ],
    "components": "list(2): [ndarray(100): [694.56425256, 714.22775937, 734.72689048, 756.10898279, 778.42471997]..., ndarray(100): [970.85148265, 1005.79811783, 1042.61499047, 1081.43391833, 1122.39812907]...]"
  },
  "x": "ndarray(100): [0.8003067, 0.80104019, 0.80177368, 0.80250717, 0.80324066]...",
  ...
}
"""