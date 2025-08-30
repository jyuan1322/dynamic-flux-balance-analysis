import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Lorentzian function
def lorentzian(x, height, center, sigma):
    return (height * sigma**2) / ((x - center)**2 + sigma**2)

# Paths (change if needed)
working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data1_13CPro1"
input_stack = os.path.join(working_dir, "Data1_13CPro1_1H.xlsx")
input_results = os.path.join(working_dir, "peak_areas_pro1_lmfit.csv")
trace_labels = ['0hr', '12hr', '36hr']

# working_dir = "/data/local/jy1008/MA-host-microbiome/XiChen_Data/TestData_V2_ProLeu/Data4_13CLeu1"
# input_stack = os.path.join(working_dir, "Data4_13CLeu1_1H.xlsx")
# input_results = os.path.join(working_dir, "peak_areas_leu1_lmfit.csv")
# trace_labels = ['0hr', '12hr', '36hr']

# Load original trace data
df = pd.read_excel(input_stack, header=None)
data = df.iloc[2:].reset_index(drop=True)
data.columns = ['ppm'] + [f'trace_{i}' for i in range(1, df.shape[1])]
data = data.astype(float)

ppm = data['ppm'].values
traces = data.drop(columns='ppm').values
n_traces = traces.shape[1]

# Load fitting results
peak_df = pd.read_csv(input_results)

# Group results by reference label
for label, group in peak_df.groupby('label'):
    plt.figure(figsize=(10, 6))
    
    # Plot all traces
    for t in range(n_traces):
        # plt.plot(ppm, traces[:, t], label=f'Trace {t+1}', alpha=0.4)
        plt.plot(ppm, traces[:, t], label=trace_labels[t], alpha=0.4)
    
    # Reconstruct and plot Lorentzian peaks for this label
    for _, row in group.iterrows():
        if not np.isnan(row['fitted_ppm']):
            center = row['fitted_ppm']
            amplitude = row['amplitude']
            sigma = row['sigma']
            height = amplitude / (np.pi * sigma)

            # Reconstruct over a small window around the peak
            window = 0.04  # same as fit window
            mask = (ppm >= center - window) & (ppm <= center + window)
            x_fit = ppm[mask]
            y_fit = lorentzian(x_fit, height, center, sigma)
            plt.plot(x_fit, y_fit, 'r-', linewidth=2)

    plt.title(f"Traces with Fitted Lorentzian Peaks: {label}")
    plt.xlabel('ppm')
    plt.ylabel('Intensity')
    plt.gca().invert_xaxis()
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()

