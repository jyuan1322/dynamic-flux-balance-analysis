import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import LorentzianModel, VoigtModel, ConstantModel
from scipy.signal import find_peaks
from att5_peak_selector import interactive_peak_selector

# working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data1_13CPro1"
# input_stack = os.path.join(working_dir, "Data1_13CPro1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
# out_csv = os.path.join(working_dir, "peak_areas_pro1_lmfit.csv")

working_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data8_13CGlc2"
input_stack = os.path.join(working_dir, "Data8_13CGlc2_1H.xlsx")
input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
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

results = []
base_fit_window = 0.04  # window half-width around reference peak

for _, ref in ref_peaks.iterrows():
    ref_ppm = ref['ppm']
    label = ref['label']
    print(f"{label} {ref_ppm}")

    for t in range(n_traces):
        np.random.seed(101)

        y = traces[:, t]
        mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
        x_data = ppm[mask]
        y_data = y[mask]

        # Check noise
        noise = np.std(y_data)
        signal = np.max(y_data) - max(0, np.min(y_data))
        print(f"len: {len(x_data)} signal: {signal} 3xnoise: {3*noise}")
        if len(x_data) < 5 or signal < 3 * noise:
            print("Skipping")
            results.append({
                'trace': t+1,
                'label': label,
                'ref_ppm': ref_ppm,
                'fitted_ppm': np.nan,
                'amplitude': np.nan,
                'sigma': np.nan,
                'area': 0,
                'peak_index': np.nan,
                'n_peaks_in_fit': 0
            })
            continue

        # Ensure ascending ppm for lmfit
        if x_data[0] > x_data[-1]:
            x_data = x_data[::-1]
            y_data = y_data[::-1]
        
        # Detect peaks inside this window to guess number of Lorentzians and initial params
        peaks_idx, _ = find_peaks(y_data, prominence=0.1 * (max(y_data) - min(y_data)))
        if len(peaks_idx) == 0:
            # fallback: assume 1 peak at ref_ppm
            peaks_idx = [np.argmax(y_data)]
        
        # --- Limit number of peaks to 10 by distance to ref_ppm ---
        # Seeing a ton of peaks is usually a sign you are fitting a noisy region.
        max_num_peaks = 5
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
            params[prefix + 'center'].set(value=x_data[peak_idx], min=x_data[0], max=x_data[-1])
            # amplitude is the area, not the height; 0.005 is a guess for sigam
            amp_guess = (y_data[peak_idx] - min(y_data)) * np.pi * 0.005
            params[prefix + 'amplitude'].set(value=amp_guess, min=0)
            # params[prefix + 'sigma'].set(value=0.005, min=0.001, max=0.06)  # HWHM
            params[prefix + 'sigma'].set(value=0.005, min=0.001, max=0.03)  # Gaussian width
            # params[prefix + 'gamma'].set(value=0.005, min=0.001, max=0.03) # Lorentzian width
        
        params['bkg_c'].set(value=min(y_data), min=min(y_data)*0.5, max=max(y_data)*0.5)
        
        try:
            result = composite_model.fit(y_data, params, x=x_data)

            nmr_fit_outfile = f"nmr_fit_{os.path.splitext(os.path.basename(input_stack))[0]}_{label}_{ref_ppm}_{t}.pdf"
            selected_peaks = interactive_peak_selector(x_data, y_data, result,
                                                    model_type="lorentzian",
                                                    ref_ppm=ref_ppm,
                                                    plot_label=label,
                                                    savepath=nmr_fit_outfile)

            # Compute areas only for selected peaks
            for i in selected_peaks:
                prefix = f"p{i}_"
                center = result.params[prefix + 'center'].value
                amplitude = result.params[prefix + "amplitude"].value
                sigma = result.params[prefix + "sigma"].value
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
                    'n_peaks_in_fit': n_peaks
                })
                print(f"Added peak: {results[-1]}")


        except Exception as e:
            print(f"Fit failed for trace {t+1}, peak {label}: {e}")
            results.append({
                'trace': t + 1,
                'label': label,
                'ref_ppm': ref_ppm,
                'fitted_ppm': np.nan,
                'amplitude': np.nan,
                'sigma': np.nan,
                'area': 0,
                'peak_index': np.nan,
                'n_peaks_in_fit': n_peaks
            })

# Save results
out_df = pd.DataFrame(results)
out_df.to_csv(out_csv, index=False)
print(out_df.head())
# return out_df
