import os
import json
import pandas as pd

input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/progress_report_figs"

# metabolite = "Isobutyrate"

records = []
for fname in os.listdir(input_dir):
    # if fname.endswith(".json") and metabolite in fname:
    if fname.endswith(".json"):
        with open(os.path.join(input_dir, fname)) as f:
            data = json.load(f)
            if isinstance(data, dict):
                records.append(data)
            else:
                records.extend(data)

df = pd.DataFrame(records)
print(df.head())
print(len(df), "rows loaded")

df_grouped = df.groupby(["trace", "label"], as_index=False)["area"].sum()
df_grouped = df_grouped.pivot(index="label", columns="trace", values="area")
df_grouped = df_grouped.T
df_grouped = df_grouped.reset_index().rename(columns={"trace": "Time"})

df_counts = (
    df.groupby(["trace", "label"], as_index=False)
      .size()
      .rename(columns={"size": "num_peaks"})
)

with pd.option_context("display.max_rows", None):
    print(df_counts)

# isobutyrate areas were derived from 3 peaks, so divide by 3
df_grouped["Isobutyrate"] = df_grouped["Isobutyrate"] / 3.0


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

    ax1.plot(corrected_times, y_mean, linewidth=2, color=color, label=f'{target_col} mean')
    ax1.fill_between(corrected_times, lower, upper,
                     color=color, alpha=0.2, label=f'{target_col} 95% CI')

    # Original data
    ax1.scatter(corrected_times, scaled_concs, color=color, s=16, label=f'{target_col} data')

metabolites = ["Formate", "Isobutyrate", "Isoleucine", "Valine"]
# colors = cm.get_cmap("tab10", len(metabolites))
colors = ["darkorange", "royalblue", "green", "purple"]
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

for i, target_col in enumerate(metabolites):
    logistic_df, corrected_times, scaled_concs = logistic_inference(df_grouped,
                                                                target_col=target_col,
                                                                exp_id="test_glc1H")
    plot_logistic_fit2(ax1, logistic_df, corrected_times, scaled_concs, target_col, color=colors[i])

# Common labels and legend
# ax2.set_xlabel('Time (hours)')
ax1.set_xlabel('Timepoint')
ax1.set_ylabel('NMR area under peaks (a.u.)')
# ax2.set_ylabel('Scaled Concentration (mMol)')
# ax1.set_title("Logistic Fits (Means + 95% CI)")
# ax2.set_title("Posterior Sample Logistic Curves")
ax1.legend()
plt.tight_layout()
plt.savefig("logistic_fits_progress_report.pdf")
plt.show()
