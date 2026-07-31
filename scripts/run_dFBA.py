import sys, os, pickle
import cobra as cb
import networkx as nx
import numpy as np
import pandas as pd
import configparser
from typing import Tuple
from scipy import integrate
from scipy.stats import norm, spearmanr
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
import matplotlib
if sys.platform == "darwin":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.drawing.nx_agraph import graphviz_layout
from cycler import cycler
import plotly.express as px
import plotly.graph_objects as go
import stan
from dFBA_JY import dFBA, MetaboliteConstraint
from dFBA_utils_JY import *

# read from config file
config = configparser.ConfigParser()
config.optionxform = str   # <-- turn off lowercasing
# config.read("config.ini")
# 13C
# config.read("config_dfba_jan302026_UGA_HRMAS_13C_Cells.ini")
# 1H mixture (fid 25)
# config.read("config_dfba_jan302026_UGA_HRMAS_13C_Cells_1H_mixture.ini")
# config.read("config_dfba_jan302026_UGA_HRMAS_13C_Cells_1H_standard2.ini")
# config.read("config/config_dfba_may222026_UGA_HRMAS_13C_Cells_1H_standard.ini")
config.read("config/config_dfba_UGA_HRMAS_13C_Cells.ini")

output_dir = config["dfba_params"]["output_dir"]
os.makedirs(output_dir, exist_ok=True)

exp_name = config["dfba_params"]["exp_name"]

def get_time_correction(csv_path, isocaproate_col="Isocaproate 0.8479", smooth_sigma=1.0, thresh=0.05, plot=False):
    """
    Load an NMR time series and determine reaction start time based on the start of 
    isocaproate production. This is a correction factor to be applied to the rest of
    the experiment.
    """
    df = pd.read_csv(csv_path)
    times = df["Time"].values
    isocaproate_concs = df[isocaproate_col].values
    # Smooth signal
    conc_smooth = gaussian_filter1d(isocaproate_concs, sigma=smooth_sigma)
    # Get derivative of conc over time
    dCdt = np.gradient(conc_smooth, times)
    # Get the max derivative, and then take the threshold as a fraction of that
    max_dCdt = np.max(dCdt)
    max_time = times[np.argmax(dCdt)]
    threshold = thresh * max_dCdt  # 10% of max
    # Find the first time where the derivative exceeds the threshold
    start_time = times[np.where(dCdt > threshold)[0][0]]
    # plot the max derivative and threshold over the concentration time series
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(times, isocaproate_concs, label='Isocaproate Concentration')
        plt.plot(times, conc_smooth, label='Smoothed Concentration', linestyle='--')
        plt.plot(times, dCdt, label='Derivative of Concentration', color='orange')
        plt.axvline(start_time, color='green', linestyle='--', label='Start Time')
        plt.axvline(max_time, color='red', linestyle='--', label='Max Derivative Time')
        plt.xlabel('Time (hours)')
        plt.ylabel('Concentration / Derivative')
        plt.title(f'Isocaproate Production Start Time {os.path.basename(csv_path)}')
        plt.legend()
        plt.tight_layout()
        plt.show()
    return start_time


# Create a function f(t) which returns a lower and upper bound for the flux at time t.
# This version calculates bounds based on a mean and std obtained directly from the
# sample data.
def logistic_inference(csv_path, target_col, exp_id, initial_concentration=None, final_concentration=None, start_time = None, isocaproate_col="Isocaproate 0.8479"):
    
    df = pd.read_csv(csv_path)
    # Correct for time offset
    if start_time is None:
        start_time = get_time_correction(csv_path, isocaproate_col, thresh=0.05, plot=False)
    corrected_times = df['Time'] - start_time

    # Scale the concentrations to mMol using the recorded initial concentration
    if initial_concentration is not None:
        scale_factor = initial_concentration / df[target_col].iloc[0]
    elif final_concentration is not None:
        scale_factor = final_concentration / df[target_col].iloc[-1]
    else:
        raise ValueError("Must provide either initial_concentration or final_concentration")
    scaled_concs = df[target_col] * scale_factor
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
    # posterior_df["D"] = posterior_df["D_sign"] * posterior_df["D_mag"]
    with open(f"stan_logistic_samples_{exp_id}_fit.pkl", "wb") as f:  # "wb" = write binary
        pickle.dump(posterior_df, f)

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


def make_logistic_deriv_fn(df_params: pd.DataFrame, ci: float = 0.95, flip_sign: bool = True,
                           bound_scale: float = 1.0, bound_widen: float = 0.0,
                           leak: float = 0.0, leak_tol: float = 1e-4,
                           scale_factor: float = 1.0):
    df = df_params.copy()

    required = ["A", "B", "C", "D"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column {col} missing from DataFrame")

    def evaluate(t: float):
        values = df.apply(
            lambda row: (row["B"] - row["A"]) / row["D"] *
                        expit((t - row["C"]) / row["D"]) *
                        (1 - expit((t - row["C"]) / row["D"])),
            axis=1
        )
        if flip_sign:
            values = -1 * values # sign flip for intake into microbes

        # Experimental: uniformly scale the flux magnitude, to test
        # sensitivity to potential concentration-estimation errors
        # (e.g. DSS reference uncertainty) for this specific reaction.
        values = values * scale_factor

        # Compute mean and confidence intervals
        mean = values.mean()
        lower = np.percentile(values, (1 - ci) / 2 * 100)
        upper = np.percentile(values, (1 + ci) / 2 * 100)

        # widen bounds around mean without crossing zero
        lower = mean - (mean - lower) * bound_scale
        upper = mean + (upper - mean) * bound_scale

        # additive widening, in the same units as the flux (mMol/hr) —
        # applied after the multiplicative bound_scale, symmetric on both sides
        lower -= bound_widen
        upper += bound_widen

        # only add a leak if both bounds are near zero
        if abs(lower) <= leak_tol and abs(upper) <= leak_tol:
            if mean >= 0:
                upper += leak
            else:
                lower -= leak

        return lower, upper

    return evaluate

time_range = config["dfba_params"]["time_range"]

# 1. Load model
# objective = "ATP_sink"
objective = config["dfba_params"]["objective"]
# modelfile = "/data/local/jy1008/MA-host-microbiome/nmr-cdiff/data/icdf843.json"
modelfile = config["dfba_params"]["modelfile"]
model = cb.io.load_json_model(modelfile)
model.objective = objective

amino_rxns = []

for rxn in model.reactions:
    # May28 no butyrate?
    if rxn.id in ['Sec_but']:
        rxn.upper_bound = 0


# model.reactions.Sec_leuL.upper_bound = 100
# model.reactions.Sec_leuL.lower_bound = -100
# model.reactions.Sec_proL.upper_bound = 100
# model.reactions.Sec_proL.lower_bound = -100


model.solver = 'glpk'

# ── PRINT AIDAN'S BOUNDS ─────────────────────────────────────────────────────
print("Bounds after Aidan's block:")
for rxn in model.reactions:
    if (rxn.id.startswith('Ex_') and rxn.id.endswith('L')) \
            or rxn.id in ['Ex_gly', 'Ex_his', 'Ex_glc', 'Ex_cysL', 'ID_357', 'ID_506', 'ID_glyamintrans']:
        print(f"  {rxn.id} ({rxn.name}): bounds=({rxn.lower_bound}, {rxn.upper_bound})")
# exit()
# ─────────────────────────────────────────────────────────────────────────────

bound_scale_test = {}
if "dfba_bound_scale_test" in config:
    bound_scale_test = {
        k: float(v) for k, v in config["dfba_bound_scale_test"].items()
        if k not in config.defaults()
    }

bound_widen_test = {}
if "dfba_bound_widen_test" in config:
    bound_widen_test = {
        k: float(v) for k, v in config["dfba_bound_widen_test"].items()
        if k not in config.defaults()
    }

constraints = {}
dfba_consts = {k:v for k, v in config["dfba_constraints"].items() if k not in config.defaults()}
for constraint, const_file in dfba_consts.items():
    print(constraint, const_file)
    lg_df = pd.read_csv(os.path.join(config["dfba_params"]["logistic_param_dir"], const_file))
    flip_sign = constraint.startswith("Ex_")  # flip sign for uptake constraints

    scale_factor = bound_scale_test.get(constraint, 1.0)
    bound_widen = bound_widen_test.get(constraint, 0.0)
    if scale_factor != 1.0:
        print(f"[dfba_bound_scale_test] Applying {scale_factor}x to {constraint}")
    if bound_widen != 0.0:
        print(f"[dfba_bound_widen_test] Widening {constraint} bounds by +/- {bound_widen}")

    flux_fn = make_logistic_deriv_fn(lg_df,
                                     ci=0.95,
                                     flip_sign=flip_sign,
                                     bound_scale=1.0,
                                     bound_widen=bound_widen,
                                     leak=1e-6,
                                     leak_tol=1e-4,
                                     scale_factor=scale_factor)
    constraints[constraint] = MetaboliteConstraint(constraint, flux_fn)

for rxn_id, constraint in constraints.items():
    print(f"\n--- {rxn_id} ---")
    for t in [0, 10, 20, 30, 40]:
        lb, ub = constraint.get_bounds(t)
        print(f"  t={t}: lb={lb:.4f}, ub={ub:.4f}")



# ─────────────────────────────────────────────────────────────────────────────
# Plot bounds of all constraint metabolites for debugging
# ─────────────────────────────────────────────────────────────────────────────
t_plot = np.linspace(
    float(config["dfba_params"].get("t_start", "0")),
    float(time_range.split(",")[1].strip()),
    300
)
"""
pdf_out = os.path.join(config["dfba_params"]["output_dir"], "constraint_bounds.pdf")
with PdfPages(pdf_out) as pdf:
    for rxn_id, constraint in constraints.items():
        lbs, ubs = [], []
        for t in t_plot:
            lb, ub = constraint.get_bounds(t)
            lbs.append(lb)
            ubs.append(ub)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_plot, lbs, label="lb", color="blue")
        ax.plot(t_plot, ubs, label="ub", color="red")
        ax.fill_between(t_plot, lbs, ubs, alpha=0.2)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_title(rxn_id)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux bound")
        ax.legend()
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

print(f"Constraint bounds saved to {pdf_out}")
"""


"""
import glob
import matplotlib.cm as cm
# the version on Github
# csv_dir = "/data/local/jy1008/MA-host-microbiome/nmr-cdiff/scripts/process/dfba_output_bounds_JY"
# pdf_out = os.path.join(config["dfba_params"]["output_dir"], "constraint_bounds_Aidan_Github.pdf")
# version shared by Aidan
csv_dir = "/data/local/jy1008/MA-host-microbiome/nmr-cdiff/scripts/process/dfba_output_bounds_JY_v2"
pdf_out = os.path.join(config["dfba_params"]["output_dir"], "constraint_bounds_Aidan_Acetate13C.pdf")

if not os.path.exists(pdf_out):
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    cmap = cm.get_cmap("tab20", len(csv_files))  # up to 20 distinct colors

    with PdfPages(pdf_out) as pdf:
        for csv_path in csv_files:
            rxn_id = os.path.splitext(os.path.basename(csv_path))[0].removeprefix("bounds_")
            df = pd.read_csv(csv_path)

            t_plot = df["time"]
            lbs = df["lower"]
            ubs = df["upper"]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t_plot, lbs, label="lb", color="blue")
            ax.plot(t_plot, ubs, label="ub", color="red")
            ax.fill_between(t_plot, lbs, ubs, alpha=0.2)
            ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
            ax.set_title(rxn_id)
            ax.set_xlabel("Time")
            ax.set_ylabel("Flux bound")
            ax.legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        # Summary page: all upper bounds overlaid, excluding Ex_glc
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, csv_path in enumerate(csv_files):
            rxn_id = os.path.splitext(os.path.basename(csv_path))[0].removeprefix("bounds_")
            print(rxn_id)
            if rxn_id in ["Ex_glc", "Ex_leuL"]:
                continue
            df = pd.read_csv(csv_path)
            ax.plot(df["time"], df["upper"], label=rxn_id, linewidth=1, color=cmap(i))

        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_title("All Upper Bounds (excluding Ex_glc, Ex_leuL)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux bound")
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    exit()
# ─────────────────────────────────────────────────────────────────────────────
"""



tracked_reactions = [
    x.strip()
    for x in config["dfba_tracked_reactions"]["ids"].split(",")
    if x.strip()
]


# 3. Run dFBA
# ID_135: proL_c --> proD_c (proline racemase)
# ID_314: proline --> 5-aminovalerate

sim = dFBA(
    model=model,
    objective=objective,
    constraints=constraints,
    # fba_method=lambda m: m.optimize(), # use pfba instead
    time_range=tuple(map(float, time_range.split(","))),
    steps_per_hour=int(config["dfba_params"]["steps_per_hour"]), # 5
    tracked_reactions = tracked_reactions,
    fva=True
)


try:
    sim.run()
except Exception as e:
    print(f"Simulation stopped early: {e}")

sim.export_results(prefix=os.path.join(config["dfba_params"]["output_dir"], f"dfba_{exp_name}"))

# plot resulting fluxes
df = sim.solution_fluxes

completed_times = sim.timecourse[:len(list(sim.fva_bounds.values())[0]["min"])]
fva_data = {}
for rxn in sim.tracked_reactions:
    fva_data[f"{rxn}_min"] = sim.fva_bounds[rxn]["min"]
    fva_data[f"{rxn}_max"] = sim.fva_bounds[rxn]["max"]

fva_df = pd.DataFrame(fva_data, index=completed_times)
df = sim.solution_fluxes.dropna(how='all').join(fva_df)

plt.rc('axes', prop_cycle=cycler('color', plt.cm.tab20.colors))
plot_raw_fluxes(df, tracked_reactions, 
                outname=os.path.join(config["dfba_params"]["output_dir"], f"dfba_flux_out_{exp_name}"), 
                model=model, plot_bounds=False)

# Grab interesting reactions
flux_df = pd.DataFrame.from_dict(sim.all_fluxes, orient='index')
flux_df.index.name = "Time"

def is_interesting_flux(series, min_peak=0.5, min_range=0.5):
    s = series.dropna().values
    if len(s) == 0:
        return False
    max_val = np.max(s)
    min_val = np.min(s)
    return (max_val >= min_peak) and ((max_val - min_val) >= min_range)

interesting_reactions = [
    rxn for rxn in flux_df.columns
    if is_interesting_flux(flux_df[rxn], min_peak=0.05, min_range=0.25)
]

bounding_reactions = list(constraints.keys())
interesting_reactions_gt2 = [
    rxn for rxn in flux_df.columns
    if is_interesting_flux(flux_df[rxn], min_peak=2.0, min_range=0.25)
]
interesting_reactions_gt2 = [rxn for rxn in interesting_reactions_gt2 if rxn not in bounding_reactions]

interesting_reactions_lt2 = [
    rxn for rxn in flux_df.columns
    if is_interesting_flux(flux_df[rxn], min_peak=0.5, min_range=0.25)
]
interesting_reactions_lt2 = [rxn for rxn in interesting_reactions_lt2
                            if rxn not in interesting_reactions_gt2
                            and rxn not in bounding_reactions]


def plot_raw_fluxes_html(flux_df, reactions, model=None, outname="raw_fluxes.html", display_plot=True):
    """
    Plot raw fluxes for specified reactions and save as HTML.
    """
    fig = go.Figure()

    for rxn in reactions:
        if rxn in flux_df.columns:
            rxn_name = ""
            if model is not None:
                rxn_obj = model.reactions.get_by_id(rxn)
                rxn_name = rxn_obj.name
            fig.add_trace(go.Scatter(
                x=flux_df.index,
                y=flux_df[rxn],
                mode='lines',
                name=f"{rxn_name} ({rxn})",
                hoverinfo='name+y',
                line=dict(width=1)
            ))

    fig.update_layout(
        title="Raw Fluxes",
        xaxis_title="Time",
        yaxis_title="Flux",
        hovermode='closest',
        showlegend=True,
        width=1000,
        height=700
    )

    fig.write_html(outname)
    if display_plot:
        fig.show()

interesting_reactions_vis = ["ID_469", "ID_366", "ID_146", "ID_321", "ID_233", "ID_53", "ID_280",
                         "HydEB", "ICCoA-DHG-EB", "ID_314", "ID_383", "BUK", "ID_326", 
                         "ATPsynth4_1", "RNF-Complex", "ID_336", "ID_575", "ID_90",
                         "Ex_glc", "Sec_ac", "Sec_alaL", "Sec_eto", "Sec_for", "Sec_lacS",
                         "Ex_proL", "Ex_leuL", "Ex_ileL", "Sec_isobuta", "Sec_isocap", "Sec_ival", "Sec_ppa"]
plot_raw_fluxes_html(flux_df, interesting_reactions_vis, model=model, display_plot=False,
                     outname=os.path.join(config["dfba_params"]["output_dir"], f'interesting_fluxes_subset_{exp_name}.html'))
# Also plot all reactions
plot_raw_fluxes_html(flux_df, interesting_reactions, model=model, display_plot=False,
                     outname=os.path.join(config["dfba_params"]["output_dir"], f'interesting_fluxes_all_{exp_name}.html'))

flux_df.to_csv(os.path.join(config["dfba_params"]["output_dir"], f'dfba_fluxes_all_{exp_name}.csv'))


# ------------------------------
# Compare to NatChemBio Figure 3
# ------------------------------

# Load data
dfall = pd.read_csv(
    os.path.join(config["dfba_params"]["output_dir"], f'dfba_{exp_name}_fluxes.csv')
)
dfall = dfall.rename(columns={"Unnamed: 0": "Time"})
dfmin = pd.read_csv(
    os.path.join(config["dfba_params"]["output_dir"], f'dfba_{exp_name}_fva_min.csv')
)
dfmin = dfmin.rename(columns={"Unnamed: 0": "Time"})
dmax = pd.read_csv(
    os.path.join(config["dfba_params"]["output_dir"], f'dfba_{exp_name}_fva_max.csv')
)
dmax = dmax.rename(columns={"Unnamed: 0": "Time"})



# Define multiple panels (each will become one PDF page)
panels = {
    "panel_a1": {
        "rxns": ["ID_90"],
        "labels": ["formate hydrogenase"],
        "flip": [],
        "colors": ["#f542ef"]
    },
    "panel_a2": {
        "rxns": ["Sec_leuL", "Sec_proL"],
        "labels": ["secretion leucine", "secretion proline"],
        "flip": [],
        "colors": ["#f542ef", "#4ef542"]
    },
    "panel_b": {
        "rxns": ["ID_469", "ID_366", "ID_146", "ID_321"],
        "labels": ["cystathionine", "isovalerate kinase",
                   "2-methylbutyrate kinase", "isobutyrate kinase"],
        "flip": [],
        "colors": ["#f542ef", "#4ef542", "#cbf542", "#257d56"]
    },
    "panel_b2": {
        "rxns": ["ID_469"],
        "labels": ["cystathionine"],
        "flip": [],
        "colors": ["#f542ef", "#4ef542", "#cbf542", "#257d56"]
    },
    "panel_b3": {
        "rxns": ["ID_146"],
        "labels": ["2-methylbutyrate kinase"],
        "flip": [],
        "colors": ["#f542ef", "#4ef542", "#cbf542", "#257d56"]
    },
    "panel_c": {
        "rxns": ["ID_233", "ID_53", "ID_280"],
        "labels": ["PGK", "PFOR", "acetate kinase"],
        "flip": ["ID_233"],
        "colors": ["#7700ff", "#f205de", "#f20505"]
    },
    "panel_d": {
        "rxns": ["HydEB"],
        "labels": ["hydrogenase"],
        "flip": ["HydEB"],
        "colors": ["#f29407"]
    },
    "panel_e": {
        "rxns": ["ICCoA-DHG-EB", "ID_314"],
        "labels": ["isocaprenoyl-CoA reductase", "Proline reductase"],
        "flip": [],
        "colors": ["#f505e9", "#a3051a"]
    },
    "panel_f": {
        "rxns": ["ID_383", "BUK"],
        "labels": ["ethanol dehydrogenase", "butyrate kinase"],
        "flip": ["ID_383"],
        "colors": ["#05f519", "#d9d904"]
    },
    "panel_g": {
        "rxns": ["ID_326"],
        "labels": ["acetyl-CoA synthetase"],
        "flip": [],
        "colors": ["#02f0ec"]
    },
    "panel_h": {
        "rxns": ["ATPsynth4_1", "RNF-Complex"],
        "labels": ["ATP synthase", "RNF complex"],
        "flip": [],
        "colors": ["#a19f9f", "#525151"]
    },
    "panel_i": {
        "rxns": ["ID_336", "ID_575"],
        "labels": ["alanine transaminase", "glutamate dehydrogenase"],
        "flip": ["ID_336"],
        "colors": ["#0f07f2", "#02d7f7"]
    },
}

# Create multipage PDF
def plot_interesting_rxns_panels(panels, dfall, dfmin, dmax, pdf_out, clip_bounds=True):
    with PdfPages(pdf_out) as pdf:
        for panel_name, panel_data in panels.items():
            rxns = panel_data["rxns"]
            labels = panel_data["labels"]
            rxn_dict = dict(zip(rxns, labels))

            plt.figure(figsize=(8, 4))

            for rxn, color in zip(rxns, panel_data.get("colors", [None] * len(rxns))):
                flip = -1 if rxn in panel_data.get("flip", []) else 1

                optimal = flip * dfall[rxn]
                fva_min = flip * dfmin[rxn]
                fva_max = flip * dmax[rxn]

                fva_lower = fva_min.combine(fva_max, min)
                fva_upper = fva_min.combine(fva_max, max)

                if clip_bounds:
                    opt_min = optimal.min()
                    opt_max = optimal.max()
                    lower_clip = opt_min * 1.05 if opt_min < 0 else opt_min / 1.05
                    upper_clip = opt_max * 1.05 if opt_max > 0 else opt_max / 1.05

                    fva_lower = fva_lower.clip(lower=lower_clip)
                    fva_upper = fva_upper.clip(upper=upper_clip)

                line, = plt.plot(dfall["Time"], optimal, label=rxn_dict[rxn], color=color)
                if rxn in dfmin.columns and rxn in dmax.columns:
                    plt.fill_between(
                        dfall["Time"],
                        fva_lower,
                        fva_upper,
                        color=line.get_color(),
                        alpha=0.2
                    )

            plt.xlabel("Time")
            plt.ylabel("Flux value")
            plt.title(panel_name)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            pdf.savefig()
            plt.close()

pdf_out = os.path.join(config["dfba_params"]["output_dir"], f'interesting_rxns_all_panels_{exp_name}.pdf')
plot_interesting_rxns_panels(panels, dfall, dfmin, dmax, pdf_out, clip_bounds=False)
pdf_out_clip = os.path.join(config["dfba_params"]["output_dir"], f'interesting_rxns_all_panels_{exp_name}_clip.pdf')
plot_interesting_rxns_panels(panels, dfall, dfmin, dmax, pdf_out_clip, clip_bounds=True)
print("Saved multi-page PDF: interesting_rxns_all_panels.pdf")