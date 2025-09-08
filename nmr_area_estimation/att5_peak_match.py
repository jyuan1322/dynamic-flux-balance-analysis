import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from lmfit import Model
from lmfit.models import LorentzianModel, VoigtModel, ConstantModel
from scipy.signal import find_peaks
from att5_peak_selector import interactive_peak_selector

# Close any existing plots (this also helps recover from Ctrl-C in previous run)
plt.close('all')

# working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data1_13CPro1"
# input_stack = os.path.join(working_dir, "Data1_13CPro1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
# out_csv = os.path.join(working_dir, "peak_areas_pro1_lmfit.csv")

working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data8_13CGlc2"
input_stack = os.path.join(working_dir, "Data8_13CGlc2_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
input_ref_peaks = os.path.join(working_dir, "cfg_1H_temp.txt")
out_csv = os.path.join(working_dir, "peak_areas_glc2_lmfit.csv")

# working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data4_13CLeu1"
# input_stack = os.path.join(working_dir, "Data4_13CLeu1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
# out_csv = os.path.join(working_dir, "peak_areas_leu1_lmfit.csv")

# Load data
# def run_fit(working_dir, input_stack, input_ref_peaks, out_csv):
df = pd.read_excel(input_stack, header=None)
data = df.iloc[2:].reset_index(drop=True)
data.columns = ['ppm'] + [f'trace_{i}' for i in range(1, df.shape[1])]
data = data.astype(float)

ppm = data['ppm'].values
traces = data.drop(columns='ppm').values
n_traces = traces.shape[1]

ref_peaks = pd.read_csv(input_ref_peaks, sep="\t", header=None, names=["ppm", "label"])

def lorentzian_area_lmfit(amplitude, sigma):
    # lmfit Lorentzian 'sigma' is HWHM (half-width at half max)
    return np.pi * amplitude * sigma


def plot_traces(data, ref_ppm, plot_title, base_fit_window=0.04):
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
        ax.plot(x_data, y_data, color=colors[i], label=f'Trace {t}')

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("ppm")
    ax.set_ylabel("Intensity")
    ax.set_title(plot_title)
    ax.invert_xaxis()
    plt.show(block=False)

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

def calculate_area(data, label, ref_ppm, t, exp_name="", base_fit_window=0.04, prominence_factor=0.1, seed=101):
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

    # Detect peaks inside this window to guess number of Lorentzians and initial params
    peaks_idx, _ = find_peaks(y_data, prominence=prominence_factor * (max(y_data) - min(y_data)))
    if len(peaks_idx) == 0:
        # fallback: assume 1 peak at ref_ppm
        peaks_idx = [np.argmax(y_data)]

    # --- Limit number of peaks to 10 by distance to ref_ppm ---
    # Seeing a ton of peaks is usually a sign you are fitting a noisy region.
    max_num_peaks = 10
    if len(peaks_idx) > max_num_peaks:
        distances = np.abs(x_data[peaks_idx] - ref_ppm)
        closest_idx = np.argsort(distances)[:max_num_peaks]  # indices of 10 closest
        peaks_idx = [peaks_idx[i] for i in closest_idx]

    # Having 2 peaks makes it easier for lmfit to fit a single bump.
    n_peaks = len(peaks_idx)
    if n_peaks == 1:
        n_peaks = n_peaks + 1

    # Build composite model of n_peaks Lorentzians + baseline
    composite_model = None
    params = None

    for i in range(n_peaks):
        prefix = f'p{i}_'
        model = LorentzianModel(prefix=prefix)
        # model = VoigtModel(prefix=prefix)
        if composite_model is None:
            composite_model = model
        else:
            composite_model += model

    # Add constant baseline as a parameter
    baseline = ConstantModel(prefix='bkg_')
    composite_model += baseline

    params = composite_model.make_params()

    # Set initial guesses
    for i, peak_idx in enumerate(peaks_idx):
        prefix = f'p{i}_'

        # Center jitter: ±0.01 around the detected peak position
        center_guess = x_data[peak_idx] + np.random.uniform(-0.01, 0.01)
        params[prefix + 'center'].set(
            value=center_guess,
            min=x_data[0],
            max=x_data[-1]
        )

        # Amplitude jitter: scale by a random factor in [0.8, 1.2]
        amp_guess = (y_data[peak_idx] - min(y_data)) * np.pi * 0.005
        amp_guess *= np.random.uniform(0.8, 1.2)
        params[prefix + 'amplitude'].set(value=amp_guess, min=0)

        # Sigma jitter: base value with noise
        sigma_guess = 0.005 * np.random.uniform(0.8, 1.2)
        params[prefix + 'sigma'].set(value=sigma_guess, min=0.001, max=0.03)

        # If Voigt is used, add randomness there too
        # gamma_guess = 0.005 * np.random.uniform(0.8, 1.2)
        # params[prefix + 'gamma'].set(value=gamma_guess, min=0.001, max=0.03)

    params['bkg_c'].set(value=min(y_data), min=min(y_data)*0.5, max=max(y_data)*0.5)

    results = []
    try:
        comp_model = composite_model.fit(y_data, params, x=x_data)

        nmr_fit_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.pdf"
        selected_peaks = interactive_peak_selector(x_data, y_data, comp_model,
                                                model_type="lorentzian",
                                                ref_ppm=ref_ppm,
                                                plot_label=label,
                                                savepath=nmr_fit_outfile)
        if len(selected_peaks) == 0:
            raise RuntimeError("No peaks selected")

        # Compute areas only for selected peaks
        total_area = 0
        for i in selected_peaks:
            prefix = f"p{i}_"
            center = comp_model.params[prefix + 'center'].value
            amplitude = comp_model.params[prefix + "amplitude"].value
            sigma = comp_model.params[prefix + "sigma"].value
            area = np.pi * amplitude * sigma
            # store into results table

            results.append({
                'trace': t + 1,
                'label': label,
                'ref_ppm': ref_ppm,
                'fitted_ppm': center,
                'amplitude': amplitude,
                'sigma': sigma,
                'area': area,
                'peak_index': i,
                'n_peaks_in_fit': n_peaks,
                'random_seed': seed
            })
            print(f"Added peak: {results[-1]}")
            total_area += area
        print(f"Total area for trace {t}, peak {label}: {total_area}")

    except Exception as e:
        print(f"Fit failed for trace {t}, peak {label}: {e}")
        results.append({
            'trace': t + 1,
            'label': label,
            'ref_ppm': ref_ppm,
            'fitted_ppm': np.nan,
            'amplitude': np.nan,
            'sigma': np.nan,
            'area': 0,
            'peak_index': np.nan,
            'n_peaks_in_fit': 0,
            'random_seed': seed
        })
    json_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.json"
    with open(json_outfile, "w") as f:
        json.dump(results, f, indent=2)
    return results


# results = []
# base_fit_window = 0.04  # window half-width around reference peak

for _, ref in ref_peaks.iterrows():
    ref_ppm = ref['ppm']
    label = ref['label']
    print(f"{label} {ref_ppm}")

    # base_fit_window = 0.04
    # prominence_factor = 0.1

    # 13C_butyrate
    # base_fit_window = 0.03
    # prominence_factor = 0.01

    # Isobutyrate
    base_fit_window = 0.03
    prominence_factor = 0.01

    plot_traces_colorbar(data, ref_ppm, plot_title=f"{label} {ref_ppm}", base_fit_window = base_fit_window)

    for t in range(n_traces):
        exp_name = os.path.splitext(os.path.basename(input_stack))[0]
        json_outfile = f"nmr_fit_{exp_name}_{label}_{ref_ppm}_{t}.json"
        if os.path.exists(json_outfile):
            print(f"Skipping trace {t}/{n_traces} for peak {label} at {ref_ppm} ppm in {exp_name}, already done.")
            with open(json_outfile, "r") as f:
                results_sub = json.load(f)
            # results.extend(results_sub)
            continue
        else:
            print("-" * 80)
            print(f"Fitting trace {t}/{n_traces} for peak {label} at {ref_ppm} ppm in {exp_name}")
            results_sub = calculate_area(data=data, label=label, ref_ppm=ref_ppm, t=t,
                                        exp_name = exp_name, base_fit_window=base_fit_window,
                                        prominence_factor=prominence_factor, seed=101)
            # results.extend(results_sub)
            print("-" * 80)

# Save results
# out_df = pd.DataFrame(results)
# out_df.to_csv(out_csv, index=False)
# print(out_df.head())
# return out_df
