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
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.drawing.nx_agraph import graphviz_layout
from cycler import cycler
import plotly.express as px
import plotly.graph_objects as go
import stan
from dFBA_JY import dFBA, MetaboliteConstraint
from dFBA_utils_JY import *


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
def logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="Isocaproate 0.8479"):
    
    df = pd.read_csv(csv_path)
    # Correct for time offset
    start_time = get_time_correction(csv_path, isocaproate_col, thresh=0.05, plot=False)
    corrected_times = df['Time'] - start_time

    # Scale the concentrations to mMol using the recorded initial concentration
    # Use these values going forward
    scale_factor = initial_concentration / df[target_col][0]
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

def plot_logistic_fit(logistic_df, corrected_times, scaled_concs):
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


def make_logistic_deriv_fn(df_params: pd.DataFrame, ci: float = 0.95):
    """
    Returns a function `evaluate(t)` that evaluates all curves
    in df_params at time t and returns the lower and upper CI.

    df_params must have columns ['A', 'B', 'C', 'D'].
    """
    df = df_params.copy()  # store internally

    required = ["A", "B", "C", "D"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column {col} missing from DataFrame")

    def evaluate(t: float):
        # Compute all logistic derivatives at t
        # values = df.apply( # This is the plain logistic, not the derivative
        #     lambda row: row["A"] + (row["B"] - row["A"]) * expit((t - row["C"]) / row["D"]),
        #     axis=1
        # )
        values = df.apply(
            lambda row: (row["B"] - row["A"]) / row["D"] *
                        expit((t - row["C"]) / row["D"]) *
                        (1 - expit((t - row["C"]) / row["D"])),
            axis=1
        )
        values = -1 * values # sign flip for intake into microbes
        # Compute mean and confidence intervals
        mean = values.mean()
        lower = np.percentile(values, (1 - ci) / 2 * 100)
        upper = np.percentile(values, (1 + ci) / 2 * 100)
        return lower, upper

    return evaluate

pro_csv_paths = [
    "concentration_estimation/Data1_13CPro1_areas.csv",
    "concentration_estimation/Data2_13CPro2_areas.csv",
    "concentration_estimation/Data3_13CPro3_areas.csv"
]
leu_csv_paths = [
    "concentration_estimation/Data4_13CLeu1_areas.csv"
]
# Pro1 time_range = (0, 48), steps_per_hour=5
# Pro2 time_range = (-7, 33), steps_per_hour=5
# Pro3 time_range = (-11, 30), steps_per_hour=5
"""
time_ranges = {
    "13CPro1": (0, 48),
    "13CPro2": (-7, 33),
    "13CPro3": (-11, 30)
}
"""
time_ranges = {
    "13CLeu1": (-12, 24)
}
# for csv_path in pro_csv_paths:
for csv_path in leu_csv_paths:
    plot_logistics = False
    # exp_id = [p for p in csv_path.split('_') if 'Pro' in p][0]
    exp_id = [p for p in csv_path.split('_') if 'Leu' in p][0]

    get_time_correction(csv_path, isocaproate_col="13C_Isocaproate 0.7453", thresh=0.05, plot=True)

target_col = "13C_Leu 1.7912"
initial_concentration = 15.0 # mMol
lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="13C_Isocaproate 0.7453")
if plot_logistics:
    plot_logistic_fit(lg_df, corrected_times, scaled_concs)
leu_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

target_col = "Tryptophan 7.5354"
initial_concentration = 0.490196078 # mMol
lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="13C_Isocaproate 0.7453")
if plot_logistics:
    plot_logistic_fit(lg_df, corrected_times, scaled_concs)
trp_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

target_col = "Isoleucine 1.2272"
initial_concentration = 2.290076336 # mMol
lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="13C_Isocaproate 0.7453")
if plot_logistics:
    plot_logistic_fit(lg_df, corrected_times, scaled_concs)
ile_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

target_col = "Cysteine 3.3220"
initial_concentration = 4.132231405 # mMol
lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="13C_Isocaproate 0.7453")
if plot_logistics:
    plot_logistic_fit(lg_df, corrected_times, scaled_concs)
cys_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

target_col = "Proline 2.0540"
initial_concentration = 6.956521739 # mMol
lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="13C_Isocaproate 0.7453")
if plot_logistics:
    plot_logistic_fit(lg_df, corrected_times, scaled_concs)
pro_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

target_col = "Histidine 7.8415"
initial_concentration = 0.64516129 # mMol
lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="13C_Isocaproate 0.7453")
if plot_logistics:
    plot_logistic_fit(lg_df, corrected_times, scaled_concs)
his_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

target_col = "Gly 3.5467"
initial_concentration = 1.333333333 # mMol
lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="13C_Isocaproate 0.7453")
if plot_logistics:
    plot_logistic_fit(lg_df, corrected_times, scaled_concs)
gly_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

target_col = "Glucose 5.2321"
initial_concentration = 27.77777778 # mMol
lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id, isocaproate_col="13C_Isocaproate 0.7453")
if plot_logistics:
    plot_logistic_fit(lg_df, corrected_times, scaled_concs)
glc_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

    """
    target_col = "Proline 4.2469"
    initial_concentration = 15.0  # mMol
    lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id)
    if plot_logistics:
        plot_logistic_fit(lg_df, corrected_times, scaled_concs)
    pro_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

    target_col = "Glucose 5.2254"
    initial_concentration = 27.77777778  # mMol
    lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id)
    if plot_logistics:
        plot_logistic_fit(lg_df, corrected_times, scaled_concs)
    glc_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

    target_col = "Valine 1.0253"
    initial_concentration = 2.564102564  # mMol
    lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id)
    if plot_logistics:
        plot_logistic_fit(lg_df, corrected_times, scaled_concs)
    val_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

    target_col = "Leucine 0.9493"
    initial_concentration = 7.633587786  # mMol
    lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id)
    if plot_logistics:
        plot_logistic_fit(lg_df, corrected_times, scaled_concs)
    leu_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)

    target_col = "Isoluecine 0.9258"
    initial_concentration = 2.290076336  # mMol
    lg_df, corrected_times, scaled_concs = logistic_inference(csv_path, target_col, initial_concentration, exp_id)
    if plot_logistics:
        plot_logistic_fit(lg_df, corrected_times, scaled_concs)
    ile_flux_fn = make_logistic_deriv_fn(lg_df, ci=0.95)
    """

# 1. Load model
objective = "ATP_sink"
modelfile = "/data/local/jy1008/MA-host-microbiome/nmr-cdiff/data/icdf843.json"
model = cb.io.load_json_model(modelfile)
model.objective = objective

    """
    constraints = {
        "Ex_proL": MetaboliteConstraint("Ex_proL", pro_flux_fn),
        "Ex_glc": MetaboliteConstraint("Ex_glc", glc_flux_fn),
        "Ex_valL": MetaboliteConstraint("Ex_valL", val_flux_fn),
        "Ex_leuL": MetaboliteConstraint("Ex_leuL", leu_flux_fn),
        "Ex_ileL": MetaboliteConstraint("Ex_ileL", ile_flux_fn)
    }
    """
constraints = {
    "Ex_proL": MetaboliteConstraint("Ex_proL", pro_flux_fn),
    "Ex_glc": MetaboliteConstraint("Ex_glc", glc_flux_fn),
    "Ex_leuL": MetaboliteConstraint("Ex_leuL", leu_flux_fn),
    "Ex_ileL": MetaboliteConstraint("Ex_ileL", ile_flux_fn),
    "Ex_trpL": MetaboliteConstraint("Ex_trpL", trp_flux_fn),
    "Ex_cysL": MetaboliteConstraint("Ex_cysL", cys_flux_fn),
    # "Ex_his": MetaboliteConstraint("Ex_his", his_flux_fn)
    "Ex_gly": MetaboliteConstraint("Ex_gly", gly_flux_fn)
}

    """
    tracked_reactions = [   
        "ATP_sink",
        "ID_575", "ID_53", "ID_326", 
        "ID_592",
        "ID_280", "ID_634",
        "Sec_co2", "Sec_ival", "Sec_nh3",
        "Ex_hco3", "Ex_h2o",
        "RNF-Complex"
    ]
    """
tracked_reactions = [
    "ATP_sink",
    "ATPsynth4_1",
    "ID_233",
    "ID_280",
    "ID_252",
    "ID_321",
    "ID_146",
    "ID_623",
    "ID_366",
    "ID_297"
]
    """
    with open("ATP_sink_reactions_list.txt", "r") as f:
        tracked_reactions = [line.strip() for line in f]
    print(tracked_reactions)
    """

    # 3. Run dFBA
    # ID_135: proL_c --> proD_c (proline racemase)
    # ID_314: proline --> 5-aminovalerate

sim = dFBA(
    model=model,
    objective=objective,
    constraints=constraints,
    time_range=time_ranges[exp_id],
    steps_per_hour=5,
    # tracked_reactions=["ATP_sink", "ID_314", "ID_135"],
    # tracked_reactions=["ATP_sink", "ID_314", "ID_135", 
    #                    "Trans_glc", "Ex_proL", "Ex_leuL", 
    #                    "Ex_valL", "Ex_ileL", "Ex_thrL", "Sec_ac", 
    #                    "Ex_alaL", "Ex_cysL", "Sec_ppa", "Sec_2abut", "ID_326"],
    # tracked_reactions=["ATP_sink", "ID_251", "ID_252", "ID_512", "ID_474", "ID_49",
    #                 "ID_280", "ID_321", "ID_146", "ID_366", "ID_1021311", "ID_1021312",
    #                 "ID_1021313", "PPAKr", "ATPsynth4_1", "BUK", "IACK", "ImzACK", "HPhACK",
    #                 "Ex_proL", "Ex_glc", "Ex_valL", "Ex_leuL", "Ex_ileL"],
    tracked_reactions = tracked_reactions,
    fva=True
)

sim.run()
sim.export_results(exp_id)

# plot resulting fluxes
df = sim.solution_fluxes

fva_data = {}
for rxn in sim.tracked_reactions:
    fva_data[f"{rxn}_min"] = sim.fva_bounds[rxn]["min"]
    fva_data[f"{rxn}_max"] = sim.fva_bounds[rxn]["max"]
fva_df = pd.DataFrame(fva_data, index=sim.timecourse)

df = df.join(fva_df)

    # reactions = ["ID_314", "ID_314_min", "ID_314_max"]
    # reactions = ["ATP_sink", "Trans_glc", "Ex_proL", "Ex_leuL", 
    #                        "Ex_valL", "Ex_ileL", "Ex_thrL", "Sec_ac", 
    #                        "Ex_alaL", "Ex_cysL", "Sec_ppa", "Sec_2abut", "ID_326"]
    # colors = plt.cm.tab20.colors  # 20 colors
    # plt.rc('axes', prop_cycle=cycler('color', plt.cm.tab20.colors))
    # plt.rc('axes', prop_cycle=cycler('color', plt.cm.tab10.colors))
    # reactions = ["Trans_glc",
    #              "Sec_ac", "Sec_ac_min", "Sec_ac_max",
    #              "Sec_ppa", "Sec_ppa_min", "Sec_ppa_max",
    #              "ID_326",
    #              "Ex_leuL",
    #              "ATP_sink"]
    # reactions = ["ATP_sink", "ID_251", "ID_252", "ID_512", "ID_474", "ID_49",
    #              "ID_280", "ID_321", "ID_146", "ID_366", "ID_1021311", "ID_1021312",
    #              "ID_1021313", "PPAKr", "ATPsynth4_1", "BUK", "IACK", "ImzACK", "HPhACK"]
plt.rc('axes', prop_cycle=cycler('color', plt.cm.tab20.colors))
# df_conc = plot_integrated_fluxes(df, reactions, initial_conc=0)
plot_raw_fluxes(df, tracked_reactions, outname=f"dfba_flux_out_{exp_id}", model=model, plot_bounds=False)

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
    if is_interesting_flux(flux_df[rxn], min_peak=0.5, min_range=0.25)
]

bounding_reactions = ["Ex_proL", "Ex_glc", "Ex_valL", "Ex_leuL", "Ex_ileL"]
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


def plot_raw_fluxes_html(flux_df, reactions, model=None, outname="raw_fluxes.html"):
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
    fig.show()

plot_raw_fluxes_html(flux_df, interesting_reactions, model=model, outname=f'interesting_fluxes_all_v4_{exp_id}.html')
plot_raw_fluxes_html(flux_df, bounding_reactions, model=model, outname=f'interesting_fluxes_v2_bounding_v4_{exp_id}.html')
plot_raw_fluxes_html(flux_df, interesting_reactions_gt2, model=model, outname=f'interesting_fluxes_v2_gt2_v4_{exp_id}.html')
plot_raw_fluxes_html(flux_df, interesting_reactions_lt2, model=model, outname=f'interesting_fluxes_v2_lt2_v4_{exp_id}.html')
