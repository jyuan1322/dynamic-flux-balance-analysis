import os
import json
import pandas as pd

# input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data7_13CGlc1"
# input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data8_13CGlc2"
# input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data9_13CGlc3"

# input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data9_13CGlc3"

input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data7_13CGlc1_1H"

# exp_name = "Data7_13CGlc1_13C"
# exp_name = "Data8_13CGlc2_13C"
# exp_name = "Data9_13CGlc3_13C"

exp_name = "Data7_13CGlc1_1H"

# def scale_to_mMol(df):
#     df = df.copy()
#     df = df.sort_values(by="Time")
#     df_mMol = df["Time"].to_frame()
#
#     initial_Glc_conc = 27.77777778 # mM
#
#     gluc_a_values = {"13C_Acetate": 14.18,
#                      "13C_Alanine": 3.19,
#                      "13C_Butyrate": 2.35,
#                      "13C_Ethanol": 4.93}
#
#     if "13C_Glucose" in df.columns:
#         df_mMol["13C_Glucose"] = df["13C_Glucose"] * (initial_Glc_conc / df["13C_Glucose"].iloc[0])
#     for metab in gluc_a_values.keys():
#         if metab in df.columns:
#             last_conc = initial_Glc_conc * df[metab].iloc[-1] / df["13C_Glucose"].iloc[0] * gluc_a_values[metab]
#             df_mMol[metab] = df[metab] * last_conc / df[metab].iloc[-1]
#
#     return df_mMol



records = []

for fname in os.listdir(input_dir):
    if fname.endswith(".json"):
        with open(os.path.join(input_dir, fname)) as f:
            data = json.load(f)
            if isinstance(data, dict):
                records.append(data)
            else:
                raise ValueError(f"{fname} is not a flat dict JSON")

# Convert list of dicts → DataFrame
df = pd.DataFrame(records)
df["total_area"] = df["total_area"].clip(lower=0)

df = df[df["experiment_name"] == exp_name]

print(df.head())
print(len(df), "rows loaded")

df_grouped = df.pivot(index="metabolite", columns="time", values="total_area")
df_grouped = df_grouped.T
df_grouped = df_grouped.reset_index().rename(columns={"time": "Time"})

# scale_to_mMol = True

if exp_name in ["Data7_13CGlc1_13C"]:
    df_grouped["13C_Glucose"] = df_grouped["13C_Glucose"] / 4.0
    df_grouped["13C_Acetate"] = df_grouped["13C_Acetate"] / 2.0
    df_grouped["13C_Lactate"] = df_grouped["13C_Lactate"] / 2.0
    df_grouped["13C_Butanol"] = df_grouped["13C_Butanol"] / 3.0
    df_grouped["13C_Ethanol"] = df_grouped["13C_Ethanol"] / 2.0
    df_grouped["13C_Alanine"] = df_grouped["13C_Alanine"] / 2.0
    df_grouped["13C_Butyrate"] = df_grouped["13C_Butyrate"] / 2.0
    # if scale_to_mMol:
elif exp_name in ["Data8_13CGlc2_13C", "Data9_13CGlc3_13C"]:
    df_grouped["13C_Glucose"] = df_grouped["13C_Glucose"] / 4.0
    df_grouped["13C_Acetate"] = df_grouped["13C_Acetate"] / 2.0
    df_grouped["13C_Lactate"] = df_grouped["13C_Lactate"] / 2.0
    # df_grouped["13C_Butanol"] = df_grouped["13C_Butanol"] / 3.0
    df_grouped["13C_Ethanol"] = df_grouped["13C_Ethanol"] / 2.0
    df_grouped["13C_Alanine"] = df_grouped["13C_Alanine"] / 2.0
    # df_grouped["13C_Butyrate"] = df_grouped["13C_Butyrate"] / 2.0
elif exp_name in ["Data7_13CGlc1_1H"]:
    df_grouped["13C_Acetate"] = df_grouped["13C_Acetate"] / 2.0
    df_grouped["13C_Alanine"] = df_grouped["13C_Alanine"] / 1.0
    df_grouped["13C_Ethanol"] = df_grouped["13C_Ethanol"] / 4.0
    df_grouped["13C_Glucose"] = df_grouped["13C_Glucose"] / 1.0
    df_grouped["13C_butyrate"] = df_grouped["13C_butyrate"] / 3.0
    df_grouped["13C_mButanol"] = df_grouped["13C_mButanol"] / 3.0
    df_grouped["2-aminobutyrate"] = df_grouped["2-aminobutyrate"] / 3.0
    df_grouped["5-aminovalerate"] = df_grouped["5-aminovalerate"] / 3.0
    df_grouped["Arginine"] = df_grouped["Arginine"] / 3.0
    df_grouped["Formate"] = df_grouped["Formate"] / 1.0
    df_grouped["Isobutyrate"] = df_grouped["Isobutyrate"] / 4.0
    df_grouped["Isocaproate"] = df_grouped["Isocaproate"] / 3.0
    df_grouped["Leucine"] = df_grouped["Leucine"] / 4.0
    df_grouped["Methionine"] = df_grouped["Methionine"] / 1.0
    df_grouped["Proline"] = df_grouped["Proline"] / 3.0
    df_grouped["Threonine"] = df_grouped["Threonine"] / 2.0
    df_grouped["Tryptophan"] = df_grouped["Tryptophan"] / 2.0
else:
    raise ValueError(f"Unknown exp_name {exp_name}")

import os, pickle
import cobra as cb
import networkx as nx
import numpy as np
import pandas as pd
from typing import Tuple
from scipy import integrate
from scipy.stats import norm, spearmanr
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.drawing.nx_agraph import graphviz_layout
from cycler import cycler
import plotly.express as px
import plotly.graph_objects as go
import stan
# from dFBA_JY import dFBA, MetaboliteConstraint
# from dFBA_utils_JY import *




# Create a function f(t) which returns a lower and upper bound for the flux at time t.
# This version calculates bounds based on a mean and std obtained directly from the
# sample data.
def logistic_inference(df_grouped, target_col, exp_id):

    df = df_grouped # pd.read_csv(csv_path)
    # Correct for time offset
    # start_time = get_time_correction(csv_path, isocaproate_col, thresh=0.05, plot=False)
    start_time = 0.0
    corrected_times = df['Time'] - start_time

    # Scale the concentrations to mMol using the recorded initial concentration
    """
    if initial_concentration is not None:
        scale_factor = initial_concentration / df[target_col].iloc[0]
    elif final_concentration is not None:
        scale_factor = final_concentration / df[target_col].iloc[-1]
    else:
        raise ValueError("Must provide either initial_concentration or final_concentration")
    scaled_concs = df[target_col] * scale_factor
    """
    scaled_concs = df[target_col]
    # Subtract minimum to normalize to 0 if there are negative values
    if(scaled_concs.min() < 0):
        scaled_concs = scaled_concs - scaled_concs.min()

    # return form pickle if it exists
    pickle_out = f"stan_logistic_samples_{exp_id}_{target_col.replace(' ', '_')}.pkl"
    pickle_out = os.path.join(input_dir, pickle_out)
    if os.path.exists(pickle_out):
        with open(pickle_out, "rb") as f:   # "rb" = read, binary mode
            logistic_df = pickle.load(f)
        return logistic_df, corrected_times, scaled_concs

    # For Stan, scale the time points going in, and then rescale them coming out
    x = corrected_times.values
    x_scale = x.max() - x.min()
    x = x / x_scale
    y = scaled_concs.values
    y_scale = y.max()
    y = y / y_scale
    N = len(x)
    slope_guess = np.sign(spearmanr(x, y).statistic)
    # time_range = x.max() - x.min()

    logistic_3p_code = (
"""
data {
    int<lower=1> N;        // number of data points
    vector[N] x;           // independent variable
    vector[N] y;           // observed values
    real D_sign;    // Sign of the slope
}
parameters {
    real<lower=0> A;       // lower asymptote
    real<lower=0> B;       // upper asymptote
    real C;                // inflection point (could be <0 post-time correction)
    real<lower=0.001> D_mag;   // slope (now strictly positive)
    //real D_sign_raw; // sign of the slope
    real<lower=0.001> sigma;   // noise standard deviation
}
transformed parameters {
    //real D = tanh(100 * D_sign_raw) * D_mag;  // D = signed slope
    real D = D_sign * D_mag;
}
model {
    // Priors
    //B ~ student_t(3, 0.5, 0.5);           // initial concentration
    //C ~ student_t(3, 0.5, 0.5);     // inflection point time
    // The choice of normal vs student_t is very important here, oddly.
    // If using student_t, for low-slope runs like Glucose or Valine,
    // the slope tends to be near-zero, creating very wide bounds.
    A ~ normal(0, 0.5);
    B ~ normal(1, 0.5);
    C ~ normal(0.5, 0.5);

    // slope D: robust prior that discourages near-zero slopes
    // target += student_t_lpdf(D | 3, 0, 1)
    //         - log(1 + exp(-abs(D))); // optional: extra repulsion from zero
    D_mag ~ student_t(3, 0, 1); // slope magnitude
    //D_sign_raw ~ normal(0, 1);  // slope sign

    sigma ~ normal(0, 0.1 * 1);            // noise std

    // Likelihood
    // inv_logit is the logistic function
    for (n in 1:N) {
        // y[n] ~ normal(B / (1 + exp(-(x[n] - C)/D)), sigma);
        //y[n] ~ normal(B * inv_logit( (x[n] - C) / (D_sign * D_mag) ), sigma);
        y[n] ~ normal(A + (B-A) * inv_logit((x[n] - C) / (D + 1e-6)), sigma);
    }
}
""")

    stan_data = {"N": N, "x": x, "y": y, "D_sign": slope_guess}

    posterior = stan.build(logistic_3p_code, data=stan_data, random_seed=12345)
    fit = posterior.sample(num_chains=4, num_samples=1000)

    posterior_df = fit.to_frame()
    print(posterior_df.head())
    # # posterior_df["D"] = posterior_df["D_sign"] * posterior_df["D_mag"]
    # with open(f"stan_logistic_samples_{exp_id}_fit.pkl", "wb") as f:  # "wb" = write binary
    #     pickle.dump(posterior_df, f)

    logistic_df = posterior_df[["A", "B", "C", "D"]].copy()
    logistic_df['A'] = logistic_df['A'] * y_scale
    logistic_df['B'] = logistic_df['B'] * y_scale
    logistic_df['C'] = logistic_df['C'] * x_scale
    logistic_df['D'] = logistic_df['D'] * x_scale

    with open(pickle_out, "wb") as f:  # "wb" = write binary
        pickle.dump(logistic_df, f)

    # return the df of sampled logistic curves
    return logistic_df, corrected_times, scaled_concs

def plot_logistic_fit(logistic_df, corrected_times, scaled_concs, target_col):
    # plot the original data and the posterior samples
    fig, (ax1, ax2) = plt.subplots(
        2, 1,          # 2 rows, 1 column
        figsize=(10, 8),
        sharex=True    # share x-axis
    )

    # Plot the posterior samples
    y_preds = []
    for i in range(logistic_df.shape[0]):
        A = logistic_df['A'].iloc[i]
        B = logistic_df['B'].iloc[i]
        C = logistic_df['C'].iloc[i]
        D = logistic_df['D'].iloc[i]
        y_fit = A + (B-A) * (1 / (1 + np.exp(-(corrected_times - C) / D)))
        y_preds.append(y_fit)
        ax2.plot(corrected_times, y_fit, color='blue', alpha=0.01)

    y_preds = np.array(y_preds)

    # Compute mean and standard error
    y_mean = np.mean(y_preds, axis=0)
    y_std = np.std(y_preds, axis=0)
    lower, upper = np.percentile(y_preds, [2.5, 97.5], axis=0)

    # Plot mean and ±SE
    ax1.plot(corrected_times, y_mean, color='red', linewidth=2, label='Mean')
    ax1.plot(corrected_times, lower, color='blue', linewidth=1, label='± 95% CI')
    ax1.plot(corrected_times, upper, color='blue', linewidth=1)
    ax1.plot(corrected_times, y_mean - y_std, color='green', linewidth=1, label='± 1 std')
    ax1.plot(corrected_times, y_mean + y_std, color='green', linewidth=1)

    # Scatter original data
    ax1.scatter(corrected_times, scaled_concs, label='Scaled Concentration Data', s=16, color='black')

    ax2.set_xlabel('Time (hours)')
    ax1.set_ylabel(f'Scaled Concentration {target_col} (mMol)')
    ax2.set_ylabel(f'Scaled Concentration {target_col} (mMol)')
    ax1.set_title(f'{target_col}')
    ax2.set_title("Posterior Sample Logistic Curves")
    ax1.legend()
    plt.tight_layout()
    plt.show()

"""
logistic_df_ft, corrected_times_ft, scaled_concs_ft = logistic_inference(df_grouped,
                                                                target_col="Formate",
                                                                exp_id="test_glc1H")
plot_logistic_fit(logistic_df_ft, corrected_times_ft, scaled_concs_ft, target_col="Formate")

logistic_df_isb, corrected_times_isb, scaled_concs_isb = logistic_inference(df_grouped,
                                                                target_col="Isobutyrate",
                                                                exp_id="test_glc1H")
plot_logistic_fit(logistic_df_isb, corrected_times_isb, scaled_concs_isb, target_col="Isobutyrate")

logistic_df_ile, corrected_times_ile, scaled_concs_ile = logistic_inference(df_grouped,
                                                                target_col="Isoleucine",
                                                                exp_id="test_glc1H")
plot_logistic_fit(logistic_df_ile, corrected_times_ile, scaled_concs_ile, target_col="Isoleucine")

logistic_df_ile, corrected_times_ile, scaled_concs_ile = logistic_inference(df_grouped,
                                                                target_col="Valine",
                                                                exp_id="test_glc1H")
plot_logistic_fit(logistic_df_ile, corrected_times_ile, scaled_concs_ile, target_col="Valine")
"""





# single plot for all samples
def plot_logistic_fit2(ax1, logistic_df, corrected_times, scaled_concs, target_col, color):
    # Posterior samples
    y_preds = []
    for i in range(logistic_df.shape[0]):
        A, B, C, D = logistic_df.loc[i, ["A", "B", "C", "D"]]
        y_fit = A + (B - A) * (1 / (1 + np.exp(-(corrected_times - C) / D)))
        y_preds.append(y_fit)
        # ax2.plot(corrected_times, y_fit, color=color, alpha=0.01)

    y_preds = np.array(y_preds)

    # Mean + 95% CI
    y_mean = np.mean(y_preds, axis=0)
    lower, upper = np.percentile(y_preds, [2.5, 97.5], axis=0)

    ax1.plot(corrected_times, y_mean, linewidth=2, color=color, label=f'{target_col} 95% CI')
    ax1.fill_between(corrected_times, lower, upper,
                     color=color, alpha=0.2, label='_nolegend_')

    # Original data
    ax1.scatter(corrected_times, scaled_concs, color=color, s=16, label='_nolegend_')

# metabolites = ["Formate", "Isobutyrate", "Isoleucine", "Valine"]
# metabolites = ["13C_butyrate", "2-aminobutyrate", "Isobutyrate", "Threonine"]
# metabolites = ["13C_butyrate", "2-aminobutyrate", "Threonine"]
metabolites = df_grouped.columns[1:].to_list()
colors = cm.get_cmap("tab10", len(metabolites))

# Get the colormap object with the specified number of colors
import matplotlib as mpl
cmap = mpl.colormaps['tab20'].resampled(len(metabolites))

# Access the list of colors from the colormap object
colors = cmap.colors

# colors = ["darkorange", "royalblue", "green", "purple"]
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

for i, target_col in enumerate(metabolites):
    print('-'*40)
    print(target_col)
    logistic_df, corrected_times, scaled_concs = logistic_inference(df_grouped,
                                                                target_col=target_col,
                                                                exp_id="test_glc1H")
    plot_logistic_fit2(ax1, logistic_df, corrected_times, scaled_concs, target_col, color=colors[i])

# Common labels and legend
# ax2.set_xlabel('Time (hours)')
# ax1.set_xlabel('Timepoint')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('NMR area under peaks (a.u.)')
# ax2.set_ylabel('Scaled Concentration (mMol)')
# ax1.set_title("Logistic Fits (Means + 95% CI)")
# ax2.set_title("Posterior Sample Logistic Curves")
ax1.legend()
plt.tight_layout()
output_trajct_fname = f"logistic_fits_{exp_name}.pdf"
plt.savefig(os.path.join(input_dir, output_trajct_fname))
plt.show()
