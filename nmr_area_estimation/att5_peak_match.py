import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import LorentzianModel
from scipy.signal import find_peaks

working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data1_13CPro1"
input_stack = os.path.join(working_dir, "Data1_13CPro1_1H.xlsx")
input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
out_csv = os.path.join(working_dir, "peak_areas_pro1_lmfit.csv")

# working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data4_13CLeu1"
# input_stack = os.path.join(working_dir, "Data4_13CLeu1_1H.xlsx")
# input_ref_peaks = os.path.join(working_dir, "cfg_1H.txt")
# out_csv = os.path.join(working_dir, "peak_areas_leu1_lmfit.csv")

# Load data
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
    
    for t in range(n_traces):
        y = traces[:, t]
        mask = (ppm >= ref_ppm - base_fit_window) & (ppm <= ref_ppm + base_fit_window)
        x_data = ppm[mask]
        y_data = y[mask]

        if len(x_data) < 5:
            # Too few points to fit reliably
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
        
        n_peaks = len(peaks_idx)
        if n_peaks == 1:
            n_peaks = n_peaks + 1
        
        # Build composite model of n_peaks Lorentzians + baseline
        composite_model = None
        params = None
        
        for i in range(n_peaks):
            prefix = f'p{i}_'
            model = LorentzianModel(prefix=prefix)
            if composite_model is None:
                composite_model = model
            else:
                composite_model += model
        
        # Add constant baseline as a parameter
        from lmfit.models import ConstantModel
        baseline = ConstantModel(prefix='bkg_')
        composite_model += baseline
        
        params = composite_model.make_params()
        
        # Set initial guesses
        for i, peak_idx in enumerate(peaks_idx):
            prefix = f'p{i}_'
            params[prefix + 'center'].set(value=x_data[peak_idx], min=x_data[0], max=x_data[-1])
            amp_guess = y_data[peak_idx] - min(y_data)
            params[prefix + 'amplitude'].set(value=amp_guess, min=0)
            # params[prefix + 'sigma'].set(value=0.005, min=0.001, max=0.06)  # HWHM
            params[prefix + 'sigma'].set(value=0.005, min=0.001, max=0.03)  # HWHM
        
        params['bkg_c'].set(value=min(y_data), min=min(y_data)*0.5, max=max(y_data)*0.5)
        
        try:
            result = composite_model.fit(y_data, params, x=x_data)
                
            # Extract the fitted peak closest to ref_ppm
            closest_idx = None
            closest_distance = float('inf')
            for i in range(n_peaks):
                prefix = f'p{i}_'
                center = result.params[prefix + 'center'].value
                distance = abs(center - ref_ppm)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_idx = i

            # Record only the closest peak
            if closest_idx is not None:
                prefix = f'p{closest_idx}_'
                center = result.params[prefix + 'center'].value
                amplitude = result.params[prefix + 'amplitude'].value
                sigma = result.params[prefix + 'sigma'].value
                area = lorentzian_area_lmfit(amplitude, sigma)
                
                results.append({
                    'trace': t + 1,
                    'label': label,
                    'ref_ppm': ref_ppm,
                    'fitted_ppm': center,
                    'amplitude': amplitude,
                    'sigma': sigma,
                    'area': area,
                    'peak_index': closest_idx + 1,
                    'n_peaks_in_fit': n_peaks
                })

            # Plot fit
            plt.figure(figsize=(6,4))
            plt.plot(x_data, y_data, 'b.', label='Data')
            plt.plot(x_data, result.best_fit, 'r-', label='Fit')
            # Plot all individual Lorentzians
            for i in range(n_peaks):
                prefix = f'p{i}_'
                model = LorentzianModel(prefix=prefix)
                comp = model.eval(params=result.params, x=x_data)
                color = 'g--' if i != closest_idx else 'm-'
                label_str = f'Peak {i+1}' if i != closest_idx else 'Closest peak'
                plt.plot(x_data, comp, color, label=label_str)

            plt.title(f'Trace {t+1} - Peaks near {label} ({ref_ppm:.3f} ppm)')
            plt.xlabel('ppm')
            plt.ylabel('Intensity')
            plt.gca().invert_xaxis()
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout()
            plt.show()

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

