import sys, json
import configparser
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
import matplotlib as mpl
if sys.platform == "darwin":
    mpl.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.drawing.nx_agraph import graphviz_layout
from cycler import cycler
import plotly.express as px
import plotly.graph_objects as go
import stan

# read from config file
config = configparser.ConfigParser()
config.optionxform = str   # <-- turn off lowercasing
# config.read("config.ini")
# 13C
# config.read("config_feb052026_UGA_HRMAS_13C_Cells.ini")
# 1H standards (fid 21)
# config.read("config_feb052026_UGA_HRMAS_13C_Cells_1H_standard.ini")
# config.read("config_jan302026_UGA_HRMAS_13C_Cells_1H_standard2.ini")
# 1H mixture (fid 25)
# config.read("config_feb052026_UGA_HRMAS_13C_Cells_1H_mixture.ini")
# May 22, 2026 exp 4
# config.read("config_may222026_UGA_HRMAS_13C_Cells_1H_standard.ini")
# config.read("config_may222026_UGA_HRMAS_13C_Cells.ini")
# Jan-Feb 2026 annotated concentrations
# config.read("config_janfeb_reference_13C_1H.ini")
# May 26, 2026 exp 5
# config.read("config_may262026_UGA_HRMAS_13C_Cells.ini")
# config.read("config/config_may262026_UGA_HRMAS_13C_Cells_1H_standard.ini")
config.read("config/config_UGA_HRMAS_13C_Cells_1H_standard.ini")
# config.read("config/config_UGA_HRMAS_13C_Cells.ini")

# METABOLITE_GROUPS = {
#     "Set 1": ["13C_Glucose", "13C_Acetate", "13C_Ethanol"],
#     "Set 2": ["13C_CO2", "13C_Lactate", "13C_Alanine"],
#     # "Set 3": ["13C_Glucose", "13C_Ethanol", "13C_Butyrate"],
# }
# METABOLITE_GROUPS = {
#     "Set 1": ["NT_Proline", "NT_5-aminovalerate", "DSS"],
#     "Set 2": ["NT_Butyrate", "NT_Isobutyrate", "13C_Acetate", "NT_Acetate"],
#     "Set 3": ["NT_Leucine", "NT_Isoleucine", "NT_Isovalerate"],
#     "Set 4": ["13C_Lactate", "13C_Pyruvate", "NT_Pyruvate"],
#     "Set 5": ["NT_Isocaproate", "13C_Formate", "NT_Formate"],
#     "Set 6": ["13C_Glucose", "Glucose", "13C_Alanine", "13C_Ethanol", "NT_Ethanol"],
#     "Set 7": ["NT_Arginine", "NT_Histidine", "NT_Methionine", "NT_Proprionate"],
#     "Set 8": ["NT_Threonine", "NT_Tryptophan", "NT_Valine", "NT_Glycine"],
# }

METABOLITE_GROUPS = {
    "Set 1": ["NT_Proline", "NT_5-aminovalerate"],
    "Set 2": ["13C_Glucose", "13C_Acetate", "13C_Ethanol", "13C_Alanine", "13C_Lactate", "13C_Pyruvate", "13C_Formate"],
    "Set 3": ["NT_Leucine", "NT_Isocaproate", "NT_Isovalerate"],
    "Set 4": ["NT_Glycine", "NT_Acetate", "NT_Ethanol"],
    "Set 5": ["NT_Threonine", "NT_Propionate"],
    "Set 6": ["NT_Valine", "NT_Isobutyrate"],
    "Set 7": ["NT_Isoleucine"],
    "Set 8": ["NT_Arginine", "NT_Histidine", "NT_Tryptophan"],
    "Set 9": ["NT_Methionine", "NT_Formate", "NT_Pyruvate"],
}



input_dir = config['trajectories']['input_dir']
output_dir = config['trajectories']['output_dir']
os.makedirs(output_dir, exist_ok=True)
exp_name = config['trajectories']['exp_name']

# ----------------------------
# <<< NEW: check for preprocessed concentrations flag
preprocessed_concs = config['trajectories'].getboolean('preprocessed_concs', fallback=False)

if preprocessed_concs:
    # Read directly from a pre-processed CSV with Time + metabolite columns
    input_stack = config['trajectories']['input_stack']
    df_grouped = pd.read_csv(os.path.join(input_dir, input_stack))
    # Ensure the time column is named "Time"
    if 'Time' not in df_grouped.columns:
        raise ValueError("Preprocessed CSV must have a 'Time' column")
    print(f"Loaded preprocessed concentrations from {input_stack}")
    print(df_grouped.head())
else:

    # Process from json

    records = []

    for fname in os.listdir(input_dir):
        if not fname.endswith(".json"):
            continue

        with open(os.path.join(input_dir, fname)) as f:
            data = json.load(f)

        metabolite = data["metabolite"]
        exp = data["experiment_name"]

        print("Processing file:", fname)

        if exp != exp_name:
            print(f"Skipping {fname} with experiment name {exp} (looking for {exp_name})")
            continue

        if "fit_results" in data:
            areas = data["fit_results"]["areas"]
            n_traces = data["fit_results"]["n_traces_total"]
        else:
            areas = data["areas"]
            n_traces = data["n_traces_total"]

        if len(areas) != n_traces:
            raise ValueError(
                f"{fname}: areas length {len(areas)} != n_traces {n_traces}"
            )

        for trace_idx, area in enumerate(areas):
            records.append({
                "experiment_name": exp,
                "metabolite": metabolite,
                "time": trace_idx,      # or trace_idx * dt if you have one
                "total_area": area
            })

    # Convert list of dicts --> DataFrame
    df = pd.DataFrame(records)
    df["total_area"] = df["total_area"].clip(lower=0)
    df = df[df["experiment_name"] == exp_name]

    print(df.head())
    print(len(df), "rows loaded")

    if df['time'].isna().any():
        df['time'] = df['trace_index']

    df_grouped = (
        df
        .groupby(["metabolite", "time"], as_index=False)["total_area"]
        .sum()
        .pivot(index="time", columns="metabolite", values="total_area")
        .reset_index()
        .rename(columns={"time": "Time"})
    )
    print(df_grouped.columns)

# --------------------
# If you need to rename a metabolite
# --------------------
# df_grouped["13C_Alanine"] = df_grouped["13C_Alanine2"] / 1.0
# df_grouped = df_grouped.drop(columns=["13C_Alanine2"])

if not preprocessed_concs:
    # rescale according to proton number
    proton_num = {k: float(v) for k, v in config["proton_num"].items() if k not in config.defaults()}
    for metabolite, protons in proton_num.items():
        if metabolite in df_grouped.columns:
            df_grouped[metabolite] = df_grouped[metabolite] / protons

    # write concentrations scaled by proton number
    os.makedirs(os.path.join(output_dir, "logistic_params"), exist_ok=True)
    df_grouped.to_csv(os.path.join(output_dir, "logistic_params",
                                   f"{exp_name}_scaled_areas_10202025.csv"), index=False)



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
    pickle_out = os.path.join(output_dir, "logistic_params", pickle_out)
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

    // sigma ~ normal(0, 0.1 * 1);            // noise std
    sigma ~ student_t(3, 0, 0.1 * 1);            // noise std

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

    logistic_df = posterior_df[["A", "B", "C", "D", "sigma"]].copy()
    logistic_df['A'] = logistic_df['A'] * y_scale
    logistic_df['B'] = logistic_df['B'] * y_scale
    logistic_df['C'] = logistic_df['C'] * x_scale
    logistic_df['D'] = logistic_df['D'] * x_scale
    # sigma is in y-space only
    logistic_df['sigma'] = logistic_df['sigma'] * y_scale

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
    y_means = []
    y_lowers = []
    y_uppers = []
    for i in range(logistic_df.shape[0]):
        A = logistic_df['A'].iloc[i]
        B = logistic_df['B'].iloc[i]
        C = logistic_df['C'].iloc[i]
        D = logistic_df['D'].iloc[i]
        sigma = logistic_df['sigma'].iloc[i]
        y_mean = A + (B-A) * (1 / (1 + np.exp(-(corrected_times.values - C) / D)))
        # y_preds.append(y_fit)
        y_means.append(y_mean)
        y_lowers.append(y_mean - 1.96 * sigma)
        y_uppers.append(y_mean + 1.96 * sigma)

        ax2.plot(corrected_times, y_mean, color='blue', alpha=0.01)

    # y_preds = np.array(y_preds)

    # Compute mean and standard error
    # y_mean = np.mean(y_preds, axis=0)
    # y_std = np.std(y_preds, axis=0)
    # lower, upper = np.percentile(y_preds, [2.5, 97.5], axis=0)
    y_means = np.mean(y_means, axis=0)
    y_lowers = np.mean(y_lowers, axis=0)
    y_uppers = np.mean(y_uppers, axis=0)

    # Plot mean and ±SE
    ax1.plot(corrected_times, y_means, color='red', linewidth=2, label='Mean')
    ax1.plot(corrected_times, y_lowers, color='blue', linewidth=1, label='± 95% CI')
    ax1.plot(corrected_times, y_uppers, color='blue', linewidth=1)
    # ax1.plot(corrected_times, y_mean - y_std, color='green', linewidth=1, label='± 1 std')
    # ax1.plot(corrected_times, y_mean + y_std, color='green', linewidth=1)

    # Scatter original data
    ax1.scatter(corrected_times, scaled_concs, label='Scaled Concentration Data', s=16, color='black')

    ax2.set_xlabel('Time (hours)')
    ax1.set_ylabel(f'Raw Area proton scaled {target_col} (a.u.)')
    ax2.set_ylabel(f'Raw Area proton scaled {target_col} (a.u.)')
    ax1.set_title(f'{target_col}')
    ax2.set_title("Posterior Sample Logistic Curves")
    ax1.legend()
    plt.tight_layout()
    plt.show()

# --------------------------------------
# plot logistic fits for all metabolites
# --------------------------------------
metabolites = [col for col in df_grouped.columns if col not in ("Time", "Samplecode")]

plot_individual = config['trajectories'].getboolean('plot_individual_metabs', fallback=True)

for col in metabolites:
    logistic_df, corrected_times, scaled_concs = logistic_inference(df_grouped,
                                                                target_col=col,
                                                                exp_id=exp_name)
    if plot_individual:
        plot_logistic_fit(logistic_df, corrected_times, scaled_concs, target_col=col)


# single plot for all samples
def plot_logistic_fit2(ax1, logistic_df, corrected_times, scaled_concs, target_col, color):
    # Posterior samples
    y_means = []
    y_lowers = []
    y_uppers = []
    # for i in range(logistic_df.shape[0]):
    #     A, B, C, D = logistic_df.loc[i, ["A", "B", "C", "D"]]
    for i in range(logistic_df.shape[0]):
        A, B, C, D = logistic_df.iloc[i][["A", "B", "C", "D"]]  # <<< .iloc not .loc
        sigma = logistic_df['sigma'].iloc[i]
        y_fit = A + (B - A) * (1 / (1 + np.exp(-(corrected_times.values - C) / D)))
        # y_preds.append(y_fit)
        y_means.append(y_fit)
        y_lowers.append(y_fit - 1.96 * sigma)
        y_uppers.append(y_fit + 1.96 * sigma)
        # ax2.plot(corrected_times, y_fit, color=color, alpha=0.01)

    # y_preds = np.array(y_preds)

    # Mean + 95% CI
    # y_mean = np.mean(y_preds, axis=0)
    # lower, upper = np.percentile(y_preds, [2.5, 97.5], axis=0)
    y_means = np.mean(y_means, axis=0)
    y_lowers = np.mean(y_lowers, axis=0)
    y_uppers = np.mean(y_uppers, axis=0)

    ax1.plot(corrected_times, y_means, linewidth=2, color=color, label=f'{target_col} 95% CI')
    ax1.fill_between(corrected_times, y_lowers, y_uppers,
                     color=color, alpha=0.2, label='_nolegend_')

    # Original data
    ax1.scatter(corrected_times, scaled_concs, color=color, s=16, label='_nolegend_')
    # logistic_pred_lists = [corrected_times, y_mean, lower, upper]
    # logistic_pred_cols = [f"{target_col}_times", f"{target_col}_mean", f"{target_col}_lower", f"{target_col}_upper"]
    # logistic_pred_df = pd.DataFrame(dict(zip(logistic_pred_cols, logistic_pred_lists)))
    logistic_pred_lists = [corrected_times.values, y_means, y_lowers, y_uppers]  # <<< .values
    logistic_pred_cols = [f"{target_col}_times", f"{target_col}_mean", f"{target_col}_lower", f"{target_col}_upper"]
    logistic_pred_df = pd.DataFrame(dict(zip(logistic_pred_cols, logistic_pred_lists)))
    return logistic_pred_df

# Get the colormap object with the specified number of colors
cmap = mpl.colormaps['tab20'].resampled(len(metabolites))
colors = cmap.colors
# revert to default colors
# colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax1 = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

all_logistic_preds = None
logistic_params = []
logistic_df_dict = {}
for i, target_col in enumerate(metabolites):
    print('-'*40)
    print(target_col)
    logistic_df, corrected_times, scaled_concs = logistic_inference(df_grouped,
                                                                target_col=target_col,
                                                                exp_id=exp_name)
    logistic_pred_df = plot_logistic_fit2(ax1, logistic_df, corrected_times, scaled_concs, target_col, color=colors[i])
    # Combine with previous results like cbind
    if all_logistic_preds is None:
        all_logistic_preds = logistic_pred_df
    else:
        all_logistic_preds = pd.concat([all_logistic_preds, logistic_pred_df], axis=1)
    logistic_params.append({
        "metab": target_col,
        "A": logistic_df["A"].mean(),
        "B": logistic_df["B"].mean(),
        "C": logistic_df["C"].mean(),
        "D": logistic_df["D"].mean()
    })
    # store logstistic_df parameters
    logistic_df_dict[target_col] = logistic_df

logistic_params = pd.DataFrame(logistic_params)
logistic_params = logistic_params.set_index("metab")


# write logistic params for metabolites prior to scaling to mMol
os.makedirs(os.path.join(output_dir, "logistic_params"), exist_ok=True)
for metab in logistic_df_dict:
    logistic_df_dict[metab].to_csv(os.path.join(output_dir, "logistic_params",
            f"logistic_params_samples_{exp_name}_{metab.replace(' ', '_')}.csv"),
            index=False)

# Common labels and legend
# ax2.set_xlabel('Time (hours)')
# ax1.set_xlabel('Timepoint')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('NMR area under peaks (a.u.)')
# ax1.set_ylabel('Scaled Concentration (mMol)')
# ax2.set_ylabel('Scaled Concentration (mMol)')
# ax1.set_title("Logistic Fits (Means + 95% CI)")
# ax2.set_title("Posterior Sample Logistic Curves")
ax1.legend()
plt.tight_layout()
output_trajct_fname = f"logistic_fits_raw_areas_{exp_name}.pdf"
# plt.savefig(os.path.join(output_dir, output_trajct_fname))
plt.show()

# save plots to multiple pdf pages
pdf_out = os.path.join(output_dir, f"logistic_fits_raw_areas_{exp_name}.pdf")
with PdfPages(pdf_out) as pdf:
    for group_name, group_metabolites in METABOLITE_GROUPS.items():
        group_metabolites = [m for m in group_metabolites if m in metabolites]
        if not group_metabolites:
            continue
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
        for target_col in group_metabolites:
            i = metabolites.index(target_col)
            line, = ax1.plot(all_logistic_preds[f"{target_col}_times"],
                     all_logistic_preds[f"{target_col}_mean"],
                     linewidth=2, label=target_col)
            color = line.get_color()
            ax1.fill_between(all_logistic_preds[f"{target_col}_times"],
                             all_logistic_preds[f"{target_col}_lower"],
                             all_logistic_preds[f"{target_col}_upper"],
                             color=color, alpha=0.2, label='_nolegend_')
            ax1.scatter(df_grouped["Time"], df_grouped[target_col],
                        color=color, s=16, label='_nolegend_')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('NMR area under peaks (a.u.)')
        ax1.set_title(group_name)
        ax1.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()
print(f"Saved: {pdf_out}")


# <<< MODIFIED: skip mMol scaling if concentrations are already preprocessed
if preprocessed_concs:
    # Concentrations are already in the correct units — use df_grouped directly
    df_grouped_conc = df_grouped.copy()
    logistic_df_dict_conc = {k: v.copy() for k, v in logistic_df_dict.items()}
    all_logistic_preds_concs = all_logistic_preds.copy()
    # Still write the logistic params (already in concentration units)
    os.makedirs(os.path.join(output_dir, "logistic_params_conc"), exist_ok=True)
    for metab in logistic_df_dict_conc:
        logistic_df_dict_conc[metab].to_csv(os.path.join(output_dir, "logistic_params_conc",
                f"logistic_params_samples_{exp_name}_{metab.replace(' ', '_')}.csv"),
                index=False)
else:
    df_grouped_conc = df_grouped.copy()

    # adjust logistic params A and B to match concentrations
    # logistic_df_dict_conc = logistic_df_dict.copy()
    logistic_df_dict_conc = { # deep copy
        k: v.copy()
        for k, v in logistic_df_dict.items()
    }

    # mean and bounds for plotting
    all_logistic_preds_concs = all_logistic_preds.copy()

    # set the initial value to the known initial concentration
    for metab, initial_conc in config["scale_mMol_to_initial"].items():
        if metab in config.defaults():
            continue
        initial_conc = float(initial_conc)
        if metab in df_grouped_conc.columns:
            initial_area = df_grouped_conc[metab].iloc[0]
            metab_const = initial_conc / initial_area
            df_grouped_conc[metab] = metab_const * df_grouped[metab]
            # adjust logistic params
            logistic_df_dict_conc[metab]["A"] = metab_const * logistic_df_dict[metab]["A"]
            logistic_df_dict_conc[metab]["B"] = metab_const * logistic_df_dict[metab]["B"]
            logistic_df_dict_conc[metab]["sigma"] = metab_const * logistic_df_dict[metab]["sigma"]  # add this
            # for plotting
            all_logistic_preds_concs[f"{metab}_mean"] = metab_const * all_logistic_preds[f"{metab}_mean"]
            all_logistic_preds_concs[f"{metab}_lower"] = metab_const * all_logistic_preds[f"{metab}_lower"]
            all_logistic_preds_concs[f"{metab}_upper"] = metab_const * all_logistic_preds[f"{metab}_upper"]

    # set the upper asymptote value (either initial or final) to the known
    # initial concentration
    for metab, initial_conc in config["scale_mMol_to_asymptote"].items():
        if metab in config.defaults():
            continue
        initial_conc = float(initial_conc)
        if metab in df_grouped_conc.columns:
            upper_asymp_value = logistic_params.loc[metab, "B"]
            metab_const = initial_conc / upper_asymp_value
            df_grouped_conc[metab] = metab_const * df_grouped[metab]
            # adjust logistic params
            logistic_df_dict_conc[metab]["A"] = metab_const * logistic_df_dict[metab]["A"]
            logistic_df_dict_conc[metab]["B"] = metab_const * logistic_df_dict[metab]["B"]
            logistic_df_dict_conc[metab]["sigma"] = metab_const * logistic_df_dict[metab]["sigma"]  # add this
            # for plotting
            all_logistic_preds_concs[f"{metab}_mean"] = metab_const * all_logistic_preds[f"{metab}_mean"]
            all_logistic_preds_concs[f"{metab}_lower"] = metab_const * all_logistic_preds[f"{metab}_lower"]
            all_logistic_preds_concs[f"{metab}_upper"] = metab_const * all_logistic_preds[f"{metab}_upper"]

    # scale glucose products using ratio method
    if "scale_mMol_to_glc_ratio" in config:
        glucose_initial_conc = df_grouped_conc["13C_Glucose"].iloc[0] # in mMol, however it was obtained
        glucose_upper_asymp = logistic_params.loc["13C_Glucose", "B"]
        for metab, ratio_slope in config["scale_mMol_to_glc_ratio"].items():
            if metab in config.defaults():
                continue
            ratio_slope = float(ratio_slope)
            if metab in df_grouped_conc.columns:
                # upper_asymp_value = logistic_params.loc[metab, "B"]
                metab_const = ratio_slope * glucose_initial_conc / glucose_upper_asymp
                df_grouped_conc[metab] = metab_const * df_grouped[metab]
                # adjust logstic params
                logistic_df_dict_conc[metab]["A"] = metab_const * logistic_df_dict[metab]["A"]
                logistic_df_dict_conc[metab]["B"] = metab_const * logistic_df_dict[metab]["B"]
                logistic_df_dict_conc[metab]["sigma"] = metab_const * logistic_df_dict[metab]["sigma"]  # add this
                # for plotting
                all_logistic_preds_concs[f"{metab}_mean"] = metab_const * all_logistic_preds[f"{metab}_mean"]
                all_logistic_preds_concs[f"{metab}_lower"] = metab_const * all_logistic_preds[f"{metab}_lower"]
                all_logistic_preds_concs[f"{metab}_upper"] = metab_const * all_logistic_preds[f"{metab}_upper"]

    # scale 1H metabolites to DSS with a fixed, known concentration
    if "scale_mMol_to_dss" in config:
        dss_known_conc = float(config["scale_mMol_to_dss"]["dss_known_conc"]) # mMol
        dss_area_mean = df_grouped["DSS"].mean()
        for metab, ratio_slope in config["scale_mMol_to_dss"].items():
            if metab in config.defaults():
                continue
            ratio_slope = float(ratio_slope)
            if metab in df_grouped_conc.columns:
                # per-timepoint scaling for raw data
                metab_const_ts = ratio_slope * dss_known_conc / df_grouped["DSS"]
                df_grouped_conc[metab] = metab_const_ts * df_grouped[metab]
                # mean scaling for logistic curves
                metab_const = ratio_slope * dss_known_conc / dss_area_mean
                logistic_df_dict_conc[metab]["A"] = metab_const * logistic_df_dict[metab]["A"]
                logistic_df_dict_conc[metab]["B"] = metab_const * logistic_df_dict[metab]["B"]
                logistic_df_dict_conc[metab]["sigma"] = metab_const * logistic_df_dict[metab]["sigma"]  # add this
                all_logistic_preds_concs[f"{metab}_mean"]  = metab_const * all_logistic_preds[f"{metab}_mean"]
                all_logistic_preds_concs[f"{metab}_lower"] = metab_const * all_logistic_preds[f"{metab}_lower"]
                all_logistic_preds_concs[f"{metab}_upper"] = metab_const * all_logistic_preds[f"{metab}_upper"]

    # write logistic params for metabolites after scaling to mMol
    os.makedirs(os.path.join(output_dir, "logistic_params_conc"), exist_ok=True)
    for metab in logistic_df_dict_conc:
        logistic_df_dict_conc[metab].to_csv(os.path.join(output_dir, "logistic_params_conc",
                f"logistic_params_samples_{exp_name}_{metab.replace(' ', '_')}.csv"),
                index=False)


# single plot for all samples - concentrations
fig, ax1 = plt.subplots(1, 1, figsize=(5, 4), sharex=True)

metabolites = [x for x in df_grouped_conc.columns if x not in ("Time", "Samplecode")]
for i, target_col in enumerate(metabolites):
    print('-'*40)
    print(target_col)
    ax1.plot(all_logistic_preds_concs[f"{target_col}_times"],
            all_logistic_preds_concs[f"{target_col}_mean"],
            linewidth=2, label=f'{target_col} 95% CI')
    ax1.fill_between(all_logistic_preds_concs[f"{target_col}_times"],
                    all_logistic_preds_concs[f"{target_col}_lower"],
                    all_logistic_preds_concs[f"{target_col}_upper"],
                    alpha=0.2, label='_nolegend_')
    ax1.scatter(df_grouped_conc["Time"], df_grouped_conc[target_col],
                s=16, label='_nolegend_')

# Common labels and legend
ax1.set_xlabel('Time (hours)')
# ax1.set_ylabel('NMR area under peaks (a.u.)')
ax1.set_ylabel('Scaled Concentration (mMol)')
# ax1.set_title("Logistic Fits (Means + 95% CI)")
ax1.legend()
plt.tight_layout()
# output_trajct_fname = f"logistic_fits_concs_{exp_name}.pdf"
# plt.savefig(os.path.join(output_dir, "logistic_params_conc", output_trajct_fname))
plt.show()

# save plots to multiple pdf pages
pdf_out = os.path.join(output_dir, "logistic_params_conc", f"logistic_fits_concs_{exp_name}.pdf")
with PdfPages(pdf_out) as pdf:
    for group_name, group_metabolites in METABOLITE_GROUPS.items():
        group_metabolites = [m for m in group_metabolites if m in metabolites]
        if not group_metabolites:
            continue
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
        for target_col in group_metabolites:
            i = metabolites.index(target_col)
            line, = ax1.plot(all_logistic_preds_concs[f"{target_col}_times"],
                     all_logistic_preds_concs[f"{target_col}_mean"],
                     linewidth=2, label=target_col)
            color = line.get_color()
            ax1.fill_between(all_logistic_preds_concs[f"{target_col}_times"],
                             all_logistic_preds_concs[f"{target_col}_lower"],
                             all_logistic_preds_concs[f"{target_col}_upper"],
                             color=color, alpha=0.2, label='_nolegend_')
            ax1.scatter(df_grouped_conc["Time"], df_grouped_conc[target_col],
                        color=color, s=16, label='_nolegend_')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Scaled Concentration (mMol)')
        ax1.set_title(group_name)
        # ax1.legend()
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
print(f"Saved: {pdf_out}")


# ----------------------------------------------------
# Sum isovalerate + isocaproate and compare to leucine
# ----------------------------------------------------
# Sum isovalerate + isocaproate and compare to leucine
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

required = ["NT_Isovalerate", "NT_Isocaproate", "NT_Leucine"]
if all(f"{m}_mean" in all_logistic_preds_concs.columns for m in required):
    
    times = all_logistic_preds_concs["NT_Leucine_times"]

    # Isovalerate individually
    line_isv, = ax.plot(times, all_logistic_preds_concs["NT_Isovalerate_mean"], linewidth=2, label="Isovalerate")
    ax.fill_between(times, all_logistic_preds_concs["NT_Isovalerate_lower"], all_logistic_preds_concs["NT_Isovalerate_upper"], alpha=0.15, color=line_isv.get_color())
    ax.scatter(df_grouped_conc["Time"], df_grouped_conc["NT_Isovalerate"], s=16, color=line_isv.get_color(), label="_nolegend_")

    # Isocaproate individually
    line_iso, = ax.plot(times, all_logistic_preds_concs["NT_Isocaproate_mean"], linewidth=2, label="Isocaproate")
    ax.fill_between(times, all_logistic_preds_concs["NT_Isocaproate_lower"], all_logistic_preds_concs["NT_Isocaproate_upper"], alpha=0.15, color=line_iso.get_color())
    ax.scatter(df_grouped_conc["Time"], df_grouped_conc["NT_Isocaproate"], s=16, color=line_iso.get_color(), label="_nolegend_")

    # Sum isovalerate + isocaproate
    combo_mean  = all_logistic_preds_concs["NT_Isovalerate_mean"]  + all_logistic_preds_concs["NT_Isocaproate_mean"]
    combo_lower = all_logistic_preds_concs["NT_Isovalerate_lower"] + all_logistic_preds_concs["NT_Isocaproate_lower"]
    combo_upper = all_logistic_preds_concs["NT_Isovalerate_upper"] + all_logistic_preds_concs["NT_Isocaproate_upper"]
    line_combo, = ax.plot(times, combo_mean, linewidth=2, label="Isovalerate + Isocaproate")
    ax.fill_between(times, combo_lower, combo_upper, alpha=0.2, color=line_combo.get_color())
    ax.scatter(df_grouped_conc["Time"],
               df_grouped_conc["NT_Isovalerate"] + df_grouped_conc["NT_Isocaproate"],
               s=16, color=line_combo.get_color(), label="_nolegend_")

    # Leucine
    line_leu, = ax.plot(times, all_logistic_preds_concs["NT_Leucine_mean"], linewidth=2, label="Leucine")
    ax.fill_between(times, all_logistic_preds_concs["NT_Leucine_lower"], all_logistic_preds_concs["NT_Leucine_upper"], alpha=0.2, color=line_leu.get_color())
    ax.scatter(df_grouped_conc["Time"], df_grouped_conc["NT_Leucine"], s=16, color=line_leu.get_color(), label="_nolegend_")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Scaled Concentration (mMol)")
    ax.set_title("Leucine vs Isovalerate + Isocaproate")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.show()
else:
    missing = [m for m in required if f"{m}_mean" not in all_logistic_preds_concs.columns]
    print(f"Missing metabolites: {missing}")