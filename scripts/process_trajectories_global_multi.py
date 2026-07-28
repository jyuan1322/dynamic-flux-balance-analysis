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

config.read("config/config_UGA_HRMAS_13C_Cells_1H_standard.ini")
# config.read("config/config_UGA_HRMAS_13C_Cells.ini")
# config.read("config/config_UGA_HRMAS_13C_Cells_temp_glc_only.ini")

# METABOLITE_GROUPS = {
#     "Set 1": ["NT_Proline", "NT_5-aminovalerate"],
#     "Set 2": ["13C_Glucose", "13C_Acetate", "13C_Ethanol", "13C_Alanine", "13C_Lactate", "13C_Pyruvate", "13C_Formate"],
#     "Set 3": ["NT_Leucine", "NT_Isocaproate", "NT_Isovalerate"],
#     "Set 4": ["NT_Glycine", "NT_Acetate", "NT_Ethanol"],
#     "Set 5": ["NT_Threonine", "NT_Propionate"],
#     "Set 6": ["NT_Valine", "NT_Isobutyrate"],
#     "Set 7": ["NT_Isoleucine"],
#     "Set 8": ["NT_Arginine", "NT_Histidine", "NT_Tryptophan"],
#     "Set 9": ["NT_Methionine", "NT_Formate", "NT_Pyruvate"],
# }
METABOLITE_GROUPS = {
    "Set 1": ["NT_Proline", "NT_5-aminovalerate"],
    "Set 2": ["NT_Leucine", "NT_Isovalerate", "NT_Isocaproate"],
    "Set 3": ["13C_Formate", "13C_Glucose", "13C_Pyruvate", "13C_Acetate", "13C_Alanine", "13C_Ethanol", "13C_Lactate"],
    "Set 6": ["NT_Isobutyrate", "NT_Valine"],
    "Set 5": ["NT_Threonine", "NT_Propionate"],
    "Set 4": ["NT_Glycine", "NT_Acetate", "NT_Ethanol"],
    "Set 7": ["NT_Formate"],
    "Set 7": ["NT_Pyruvate"],
    "Set 9": ["NT_Methionine", "NT_Threonine", "NT_Propionate"],
}
METABOLITE_COLORS = {
    "NT_Proline":          "#000080",
    "NT_5-aminovalerate":  "#8E0041",
    "NT_Leucine":          "#008080",
    "NT_Isovalerate":      "#4BDF2E",
    "NT_Isocaproate":      "#FF7674",
    "13C_Formate":         "#F78B3B",
    "13C_Glucose":         "#372FCA",
    "13C_Pyruvate":        "#F788FF",
    "13C_Acetate":         "#F5195A",
    "13C_Alanine":         "#60A5FF",
    "13C_Ethanol":         "#46B20F",
    "13C_Lactate":         "#A23E00",
    "NT_Isobutyrate":      "#FF5722",
    "NT_Valine":           "#D500F9",
    "NT_Threonine":        "#8E24AA",
    "NT_Propionate":       "#00BCD4",
    "NT_Glycine":          "#2E7D32",
    "NT_Acetate":          "#D81B60",
    "NT_Ethanol":          "#FFA000",
    "NT_Formate":          "#0D7359",
    "NT_Pyruvate":         "#1565C0",
    "NT_Methionine":       "#B8860B",
    "NT_Isoleucine":       "#9edae5",
    "NT_Arginine":         "#393b79",
    "NT_Histidine":        "#637939",
    "NT_Tryptophan":       "#8c6d31",
}


input_dir = config['trajectories']['input_dir']
output_dir = config['trajectories']['output_dir']
os.makedirs(output_dir, exist_ok=True)
exp_name = config['trajectories']['exp_name']

# ----------------------------
# Check for preprocessed concentrations flag
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



def estimate_initial_slope_sign(x, y, frac=0.5):
    """Sign of the linear trend over the first `frac` of the trajectory,
    used to fix D1's sign (and D2's sign, which is forced opposite)."""
    n = max(3, int(len(x) * frac))
    slope = np.polyfit(x[:n], y[:n], 1)[0]
    return 1.0 if slope >= 0 else -1.0

# Create a function f(t) which returns a lower and upper bound for the flux at time t.
# This version calculates bounds based on a mean and std obtained directly from the
# sample data.
def logistic_inference(df_grouped, target_col, exp_id, tau2=0.1):

    df = df_grouped
    start_time = 0.0
    corrected_times = df['Time'] - start_time

    scaled_concs = df[target_col]
    if(scaled_concs.min() < 0):
        scaled_concs = scaled_concs - scaled_concs.min()

    pickle_out = f"stan_logistic_signedsparse_samples_{exp_id}_{target_col.replace(' ', '_')}.pkl"
    pickle_out = os.path.join(output_dir, "logistic_params", pickle_out)
    if not overwrite_pkls and os.path.exists(pickle_out):
        with open(pickle_out, "rb") as f:
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

    d1_sign = estimate_initial_slope_sign(x, y)

    logistic_signed_sparse_code = (
"""
data {
    int<lower=1> N;
    vector[N] x;
    vector[N] y;
    real D1_sign;        // sign of first transition's slope, supplied as data
    real<lower=0> tau2;  // shrinkage scale for second transition's amplitude
}
parameters {
    real A;                        // baseline level
    real<lower=0> amp1;                     // first transition amplitude (unconstrained, unshrunk)
    real amp2_raw;
    real<lower=0.001> lambda2;     // local shrinkage scale
    real C1;
    real C2;
    real<lower=0.001> D1_mag;
    real<lower=0.001> D2_mag;
    real<lower=0.001> sigma;
}
transformed parameters {
    real D1 = D1_sign * D1_mag;
    real D2 = -D1_sign * D2_mag;   // fixed opposite sign, not a free parameter
    real amp2 = amp2_raw * lambda2;
}
model {
    A ~ normal(0, 0.5);
    amp1 ~ normal(0, 0.5);              // no sparsity: first transition always allowed
    amp2_raw ~ normal(0, 1);
    lambda2 ~ normal(0, tau2);    // half-normal, shrinks amp2 toward 0
    C1 ~ normal(0.5, 0.5);
    C2 ~ normal(0.5, 0.5);
    D1_mag ~ student_t(3, 0, 1);
    D2_mag ~ student_t(3, 0, 1);
    sigma ~ student_t(3, 0, 0.1);

    for (n in 1:N) {
        y[n] ~ normal(
            A + amp1 * inv_logit((x[n] - C1) / D1)
              + amp2 * inv_logit((x[n] - C2) / D2),
            sigma
        );
    }
}
""")

    stan_data = {"N": N, "x": x, "y": y, "D1_sign": d1_sign, "tau2": tau2}

    posterior = stan.build(logistic_signed_sparse_code, data=stan_data, random_seed=12345)
    fit = posterior.sample(num_chains=4, num_samples=1000)

    posterior_df = fit.to_frame()
    print(posterior_df.head())

    logistic_df = posterior_df[["A", "amp1", "amp2", "C1", "C2", "D1", "D2", "sigma"]].copy()
    logistic_df['A']     = logistic_df['A']     * y_scale
    logistic_df['amp1']  = logistic_df['amp1']  * y_scale
    logistic_df['amp2']  = logistic_df['amp2']  * y_scale
    logistic_df['C1']    = logistic_df['C1']    * x_scale
    logistic_df['C2']    = logistic_df['C2']    * x_scale
    logistic_df['D1']    = logistic_df['D1']    * x_scale
    logistic_df['D2']    = logistic_df['D2']    * x_scale
    logistic_df['sigma'] = logistic_df['sigma'] * y_scale

    with open(pickle_out, "wb") as f:
        pickle.dump(logistic_df, f)

    return logistic_df, corrected_times, scaled_concs

def plot_logistic_fit(logistic_df, corrected_times, scaled_concs, target_col, pdf=None, show=True):
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(10, 11),
        sharex=True
    )

    y_means = []
    y_lowers = []
    y_uppers = []
    comp1_means = []   # amp1 * sigmoid(C1, D1) contribution, per sample
    comp2_means = []   # amp2 * sigmoid(C2, D2) contribution, per sample
    for i in range(logistic_df.shape[0]):
        A = logistic_df['A'].iloc[i]
        amp1 = logistic_df['amp1'].iloc[i]
        amp2 = logistic_df['amp2'].iloc[i]
        C1 = logistic_df['C1'].iloc[i]
        C2 = logistic_df['C2'].iloc[i]
        D1 = logistic_df['D1'].iloc[i]
        D2 = logistic_df['D2'].iloc[i]
        sigma = logistic_df['sigma'].iloc[i]
        t = corrected_times.values

        comp1 = amp1 / (1 + np.exp(-(t - C1) / D1))
        comp2 = amp2 / (1 + np.exp(-(t - C2) / D2))
        y_mean = A + comp1 + comp2

        y_means.append(y_mean)
        y_lowers.append(y_mean - 1.96 * sigma)
        y_uppers.append(y_mean + 1.96 * sigma)
        comp1_means.append(comp1)
        comp2_means.append(comp2)

        ax2.plot(corrected_times, y_mean, color='blue', alpha=0.01)

    y_means = np.mean(y_means, axis=0)
    y_lowers = np.mean(y_lowers, axis=0)
    y_uppers = np.mean(y_uppers, axis=0)
    comp1_mean = np.mean(comp1_means, axis=0)
    comp1_lower = np.percentile(comp1_means, 2.5, axis=0)
    comp1_upper = np.percentile(comp1_means, 97.5, axis=0)
    comp2_mean = np.mean(comp2_means, axis=0)
    comp2_lower = np.percentile(comp2_means, 2.5, axis=0)
    comp2_upper = np.percentile(comp2_means, 97.5, axis=0)

    ax1.plot(corrected_times, y_means, color='red', linewidth=2, label='Mean')
    ax1.plot(corrected_times, y_lowers, color='blue', linewidth=1, label='± 95% CI')
    ax1.plot(corrected_times, y_uppers, color='blue', linewidth=1)
    ax1.scatter(corrected_times, scaled_concs, label='Scaled Concentration Data', s=16, color='black')

    # New panel: the two logistic components separately
    ax3.plot(corrected_times, comp1_mean, color='green', linewidth=2, label='Transition 1 (amp1)')
    ax3.fill_between(corrected_times, comp1_lower, comp1_upper, color='green', alpha=0.2)
    ax3.plot(corrected_times, comp2_mean, color='purple', linewidth=2, label='Transition 2 (amp2)')
    ax3.fill_between(corrected_times, comp2_lower, comp2_upper, color='purple', alpha=0.2)
    ax3.axhline(0, color='gray', linewidth=0.8, linestyle='--')

    ax2.set_xlabel('Time (hours)')
    ax3.set_xlabel('Time (hours)')
    ax1.set_ylabel(f'Raw Area proton scaled {target_col} (a.u.)')
    ax2.set_ylabel(f'Raw Area proton scaled {target_col} (a.u.)')
    ax3.set_ylabel('Component contribution (a.u.)')
    ax1.set_title(f'{target_col}')
    ax2.set_title("Posterior Sample Logistic Curves")
    ax3.set_title("Transition 1 vs Transition 2 Contributions")
    ax1.legend()
    ax3.legend()
    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(fig)
    if show:
        plt.show()
    plt.close(fig)

# --------------------------------------
# plot logistic fits for all metabolites
# --------------------------------------
metabolites = [col for col in df_grouped.columns if col not in ("Time", "Samplecode")]

plot_individual = config['trajectories'].getboolean('plot_individual_metabs', fallback=True)
overwrite_pkls = config['trajectories'].getboolean('overwrite_pkls', fallback=False)

pdf_individual_out = os.path.join(output_dir, "logistic_params", f"logistic_fits_individual_{exp_name}.pdf")
with PdfPages(pdf_individual_out) as pdf:
    for col in metabolites:
        logistic_df, corrected_times, scaled_concs = logistic_inference(df_grouped,
                                                                    target_col=col,
                                                                    exp_id=exp_name)
        # if plot_individual:
        plot_logistic_fit(logistic_df, corrected_times, scaled_concs,
                            target_col=col, pdf=pdf, show=plot_individual)
print(f"Saved: {pdf_individual_out}")


# single plot for all samples
def plot_logistic_fit2(ax1, logistic_df, corrected_times, scaled_concs, target_col, color):
    # Posterior samples
    y_means = []
    y_lowers = []
    y_uppers = []
    # for i in range(logistic_df.shape[0]):
    #     A, B, C, D = logistic_df.loc[i, ["A", "B", "C", "D"]]
    for i in range(logistic_df.shape[0]):
        A = logistic_df['A'].iloc[i]
        amp1 = logistic_df['amp1'].iloc[i]
        amp2 = logistic_df['amp2'].iloc[i]
        C1 = logistic_df['C1'].iloc[i]
        C2 = logistic_df['C2'].iloc[i]
        D1 = logistic_df['D1'].iloc[i]
        D2 = logistic_df['D2'].iloc[i]
        sigma = logistic_df['sigma'].iloc[i]
        t = corrected_times.values
        y_mean = A + amp1 / (1 + np.exp(-(t - C1) / D1)) + amp2 / (1 + np.exp(-(t - C2) / D2))
        # y_preds.append(y_fit)
        y_means.append(y_mean)
        y_lowers.append(y_mean - 1.96 * sigma)
        y_uppers.append(y_mean + 1.96 * sigma)
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
    if all_logistic_preds is None:
        all_logistic_preds = logistic_pred_df
    else:
        all_logistic_preds = pd.concat([all_logistic_preds, logistic_pred_df], axis=1)
    logistic_params.append({
        "metab": target_col,
        "A": logistic_df["A"].mean(),
        "amp1": logistic_df["amp1"].mean(),
        "amp2": logistic_df["amp2"].mean(),
        "C1": logistic_df["C1"].mean(),
        "C2": logistic_df["C2"].mean(),
        "D1": logistic_df["D1"].mean(),
        "D2": logistic_df["D2"].mean(),
        # stand-in for the old "B" (upper asymptote / final level)
        "final_level": logistic_df["A"].mean() + logistic_df["amp1"].mean() + logistic_df["amp2"].mean(),
    })
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


# ============================================================
# Build scaling factors directly from RAW data (no fit dependency,
# since only using scale_mMol_to_initial / scale_mMol_to_dss)
# ============================================================

if preprocessed_concs:
    df_grouped_conc = df_grouped.copy()
else:
    scale_factors = pd.DataFrame(1.0, index=df_grouped.index, columns=metabolites)

    if "scale_mMol_to_initial" in config:
        for metab, initial_conc in config["scale_mMol_to_initial"].items():
            if metab in config.defaults():
                continue
            if metab in df_grouped.columns:
                initial_area = df_grouped[metab].iloc[0]
                scale_factors[metab] = float(initial_conc) / initial_area

    if "scale_mMol_to_dss" in config:
        dss_known_conc = float(config["scale_mMol_to_dss"]["dss_known_conc"])
        for metab, ratio_slope in config["scale_mMol_to_dss"].items():
            if metab in config.defaults():
                continue
            if metab in df_grouped.columns:
                scale_factors[metab] = (float(ratio_slope) * dss_known_conc
                                         / df_grouped["DSS"])

    df_grouped_conc = df_grouped.copy()
    for metab in metabolites:
        df_grouped_conc[metab] = scale_factors[metab] * df_grouped[metab]

    os.makedirs(os.path.join(output_dir, "logistic_params_conc"), exist_ok=True)
    df_grouped_conc.to_csv(os.path.join(output_dir, "logistic_params_conc",
                                         f"{exp_name}_scaled_concs.csv"), index=False)


# single plot for all samples - concentrations
fig, ax1 = plt.subplots(1, 1, figsize=(5, 4), sharex=True)

metabolites_conc = [x for x in df_grouped_conc.columns if x not in ("Time", "Samplecode")]
cmap_conc = mpl.colormaps['tab20'].resampled(len(metabolites_conc))
colors_conc = cmap_conc.colors

all_logistic_preds_concs = None
logistic_df_dict_conc = {}
for i, target_col in enumerate(metabolites_conc):
    print('-'*40)
    print(target_col)
    logistic_df_conc, corrected_times_conc, scaled_concs_conc = logistic_inference(
        df_grouped_conc, target_col=target_col, exp_id=f"{exp_name}_conc"
    )
    logistic_pred_df_conc = plot_logistic_fit2(ax1, logistic_df_conc, corrected_times_conc,
                                                scaled_concs_conc, target_col, color=colors_conc[i])
    if all_logistic_preds_concs is None:
        all_logistic_preds_concs = logistic_pred_df_conc
    else:
        all_logistic_preds_concs = pd.concat([all_logistic_preds_concs, logistic_pred_df_conc], axis=1)
    logistic_df_dict_conc[target_col] = logistic_df_conc

ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Scaled Concentration (mMol)')
ax1.legend()
plt.tight_layout()
plt.show()

os.makedirs(os.path.join(output_dir, "logistic_params_conc"), exist_ok=True)
for metab in logistic_df_dict_conc:
    logistic_df_dict_conc[metab].to_csv(os.path.join(output_dir, "logistic_params_conc",
            f"logistic_params_samples_{exp_name}_{metab.replace(' ', '_')}.csv"),
            index=False)

# save plots to multiple pdf pages
pdf_out = os.path.join(output_dir, "logistic_params_conc", f"logistic_fits_concs_{exp_name}.pdf")
with PdfPages(pdf_out) as pdf:
    for group_name, group_metabolites in METABOLITE_GROUPS.items():
        group_metabolites = [m for m in group_metabolites if m in metabolites_conc]
        if not group_metabolites:
            continue
        fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
        for target_col in group_metabolites:
            color = METABOLITE_COLORS.get(target_col, "black")  # fallback if a metabolite is missing from the dict

            ax1.plot(all_logistic_preds_concs[f"{target_col}_times"],
                     all_logistic_preds_concs[f"{target_col}_mean"],
                     linewidth=2, label=target_col, color=color)
            ax1.fill_between(all_logistic_preds_concs[f"{target_col}_times"],
                             all_logistic_preds_concs[f"{target_col}_lower"],
                             all_logistic_preds_concs[f"{target_col}_upper"],
                             color=color, alpha=0.2, label='_nolegend_')
            ax1.scatter(df_grouped_conc["Time"], df_grouped_conc[target_col],
                        color=color, s=16, label='_nolegend_')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Scaled Concentration (mMol)')
        ax1.set_title(group_name)
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