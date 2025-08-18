import os, pickle
import cobra as cb
import networkx as nx
import numpy as np
import pandas as pd
from typing import Tuple
from scipy import integrate
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
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

"""
def plot_adjusted_flux(corrected_times, scaled_concs, flux_fn, conc_spline, exp_name="", target_col=""):
    t_fit = np.linspace(corrected_times.min(), corrected_times.max(), 1000)
    # Evaluate flux_fn at each t
    bounds = [flux_fn(t) for t in t_fit]

    # Unpack into lower and upper bound arrays
    lower_bounds, upper_bounds = zip(*bounds)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    # lb and lb are mean +/- margin
    means = (lower_bounds + upper_bounds) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Experimental data
    ax1.scatter(corrected_times, scaled_concs, label="Experimental Concentrations", s=16, color='black')
    ax1.plot(t_fit, conc_spline(t_fit), label='Smoothed Spline', color='blue')
    ax1.set_ylabel(f'Conc {exp_name} {target_col} (mMol)')
    ax1.legend(loc='upper right')

    # Flux CI bounds
    ax2.fill_between(t_fit, lower_bounds, upper_bounds, color='red', alpha=0.3, label='Flux ± 0.95CI')

    # Mean Flux
    ax2.plot(t_fit, means, label='Mean Flux', color='red')

    # Also plot the finite difference estimates at each point
    t = corrected_times.to_numpy()
    c = scaled_concs.to_numpy()
    dt = np.diff(t)
    dc = np.diff(c)
    # Centered finite differences (size N-2), could be padded later
    finite_deriv = np.empty_like(c)
    finite_deriv[1:-1] = (c[2:] - c[:-2]) / (t[2:] - t[:-2])
    # Forward/backward for edges
    finite_deriv[0] = (c[1] - c[0]) / (t[1] - t[0])
    finite_deriv[-1] = (c[-1] - c[-2]) / (t[-1] - t[-2])
    # Flip the sign of the finite derivative to match flux direction
    finite_deriv = -finite_deriv
    ax2.plot(corrected_times, finite_deriv, 'o', label='Finite Difference dC/dt', markersize=4, color='blue')

    ax2.set_xlabel('Time (hrs)')
    ax2.set_ylabel(f'Flux {exp_name} {target_col} (mMol/hr)')
    ax2.legend(loc='upper right')

    # save experiment info for later plotting
    exp_flux_specs = {
        "corrected_times": corrected_times,
        "scaled_concs": scaled_concs,
        "conc_spline": conc_spline,
        "target_col": target_col,
        "exp_name": exp_name,
        "t_fit": t_fit,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "means": means,
        "finite_deriv": finite_deriv
    }
    pickle.dump(exp_flux_specs, open(f"{exp_name}_{target_col.replace(' ', '_')}_flux_specs.pkl", "wb"))

    plt.tight_layout()
    plt.show()
"""

"""
# plot multiple experiments together to test the time shift
def plot_adjusted_fluxes_multiple(pkl_files):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for pkl_file in pkl_files:
        specs = pickle.load(open(pkl_file, "rb"))
        exp_name = specs["exp_name"]
        target_col = specs["target_col"]
        corrected_times = specs["corrected_times"]
        scaled_concs = specs["scaled_concs"]
        conc_spline = specs["conc_spline"]
        t_fit = specs["t_fit"]
        lower_bounds = specs["lower_bounds"]
        upper_bounds = specs["upper_bounds"]
        means = specs["means"]
        finite_deriv = specs["finite_deriv"]
        conc_spline_derivative = conc_spline.derivative()

        ax1.plot(corrected_times, conc_spline(corrected_times), label=f'{exp_name} {target_col}')
        ax1.set_ylabel(f'Conc {exp_name} {target_col} (mMol)')
        ax1.legend(loc='upper right')
        ax2.plot(t_fit, means, label=f"{exp_name} {target_col}")
        ax2.set_xlabel('Time (hrs)')
        ax2.set_ylabel(f'Flux {exp_name} {target_col} (mMol/hr)')
        ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
out_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/concentration_estimation/results_V2"
for metabolite in ["Proline", "Glucose", "Valine", "Leucine", "Isoluecine"]:
    # Find all files that contain the metabolite name
    files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if metabolite in f]
    files = sorted(files)
    plot_adjusted_fluxes_multiple(files)
"""

# Create a function f(t) which returns a lower and upper bound for the flux at time t.
# This version calculates bounds based on a mean and std obtained directly from the
# sample data.
def flux_function_bayesian(target_col, csv_path, initial_concentration, smoothing_factor=0.5,
                           flux_bounds_window = 5, flux_bounds_sigma = 1, plot=False):
    df = pd.read_csv(csv_path)
    # Correct for time offset
    start_time = get_time_correction(csv_path, thresh=0.05, plot=False)
    corrected_times = df['Time'] - start_time

    # Scale the concentrations to mMol using the recorded initial concentration
    # Use these values going forward
    scale_factor = initial_concentration / df[target_col][0]
    scaled_concs = df[target_col] * scale_factor
    # Subtract minimum to normalize to 0 if there are negative values
    if(scaled_concs.min() < 0):
        scaled_concs = scaled_concs - scaled_concs.min()


def logistic_inference(csv_path, target_col, initial_concentration, plot=False):

    df = pd.read_csv(csv_path)
    # Correct for time offset
    start_time = get_time_correction(csv_path, thresh=0.05, plot=False)
    corrected_times = df['Time'] - start_time

    # Scale the concentrations to mMol using the recorded initial concentration
    # Use these values going forward
    scale_factor = initial_concentration / df[target_col][0]
    scaled_concs = df[target_col] * scale_factor
    # Subtract minimum to normalize to 0 if there are negative values
    if(scaled_concs.min() < 0):
        scaled_concs = scaled_concs - scaled_concs.min()

    # scale the time points going in, and then rescale them coming out
    x = corrected_times.values
    x_scale = x.max() - x.min()
    x = x / x_scale
    y = scaled_concs.values
    y_scale = y.max()
    y = y / y_scale
    N = len(x)
    # time_range = x.max() - x.min()

    logistic_3p_code = (
"""
data {
    int<lower=1> N;        // number of data points
    vector[N] x;           // independent variable
    vector[N] y;           // observed values
}
parameters {
    real<lower=0> B;       // upper asymptote
    real C;                // inflection point (could be <0 post-time correction)
    real<lower=0.01> D_mag;   // slope (could be increasing or decreasing)
    real D_sign_raw; // sign of the slope
    real<lower=0.001> sigma;   // noise standard deviation
}
transformed parameters {
    real D = tanh(D_sign_raw) * D_mag;  // D = signed slope
}
model {
    // Priors
    //B ~ student_t(3, 0.5, 0.5);           // initial concentration
    //C ~ student_t(3, 0.5, 0.5);     // inflection point time
    B ~ normal(1, 0.5);
    C ~ normal(0.5, 0.5);

    // slope D: robust prior that discourages near-zero slopes
        target += student_t_lpdf(D | 3, 0, 1) 
                - log(1 + exp(-abs(D))); // optional: extra repulsion from zero

    D_sign_raw ~ normal(0, 1);  // slope sign
    sigma ~ normal(0, 0.1 * 1);            // noise std

    // Likelihood
    for (n in 1:N) {
        // y[n] ~ normal(B / (1 + exp(-(x[n] - C)/D)), sigma);
        // https://mc-stan.org/math/namespacestan_1_1math_a91d68af3b629d4048e4044c7db23a1dc.html
        //y[n] ~ normal(B * inv_logit( (x[n] - C) / (D_sign * D_mag) ), sigma);
        y[n] ~ normal(B * inv_logit((x[n] - C) / D), sigma);
    }
}
""")

    stan_data = {"N": N, "x": x, "y": y}

    posterior = stan.build(logistic_3p_code, data=stan_data, random_seed=12345)
    fit = posterior.sample(num_chains=4, num_samples=1000)

    posterior_df = fit.to_frame()
    print(posterior_df.head())
    # posterior_df["D"] = posterior_df["D_sign"] * posterior_df["D_mag"]

    if plot:
        # plot the original data and the posterior samples
        fig, (ax1, ax2) = plt.subplots(
            2, 1,          # 2 rows, 1 column
            figsize=(10, 8),
            sharex=True    # share x-axis
        )

        # Plot the posterior samples
        y_preds = []
        for i in range(posterior_df.shape[0]):
            B = posterior_df['B'].iloc[i] * y_scale
            C = posterior_df['C'].iloc[i] * x_scale
            D = posterior_df['D'].iloc[i] * x_scale
            y_fit = B * (1 / (1 + np.exp(-(corrected_times - C) / D)))
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


pro_csv_paths = [
    "concentration_estimation/Data1_13CPro1_areas.csv",
    "concentration_estimation/Data2_13CPro2_areas.csv",
    "concentration_estimation/Data3_13CPro3_areas.csv"
]
for csv_path in pro_csv_paths:
    target_col = "Proline 4.2469"
    initial_concentration = 15.0  # mMol
    logistic_inference(csv_path, target_col, initial_concentration, plot=True)

    target_col = "Glucose 5.2254"
    initial_concentration = 27.77777778  # mMol
    logistic_inference(csv_path, target_col, initial_concentration, plot=True)

    target_col = "Valine 1.0253"
    initial_concentration = 2.564102564  # mMol
    logistic_inference(csv_path, target_col, initial_concentration, plot=True)

    target_col = "Leucine 0.9493"
    initial_concentration = 7.633587786  # mMol
    logistic_inference(csv_path, target_col, initial_concentration, plot=True)

    target_col = "Isoluecine 0.9258"
    initial_concentration = 2.290076336  # mMol
    logistic_inference(csv_path, target_col, initial_concentration, plot=True)




'''
    # To estimate the mean flux, precompute the interpolation spline and its derivative
    spline = UnivariateSpline(corrected_times, scaled_concs, s=smoothing_factor)
    spline_derivative = spline.derivative()
    # To estimate the CI of the flux, calculate point-wise derivatives in a window around t
    # ci = 0.95  # Confidence interval
    # z = norm.ppf((1 + ci) / 2)
    z = 1.0 # just use 1 std for now

    min_flux_ci = 0.001
    def flux_fn(t):
        """
        Returns a lower and upper bound for the flux at time t.
        This uses bootstrapped interpolated curves to estimate the bounds.
        """
        flux_est = spline_derivative(t)
        # Get points within the window around t
        mask = np.abs(corrected_times - t) <= flux_bounds_window
        # NOTE: want position-based indexing in numpy, not label-based in pandas
        local_times = corrected_times[mask].to_numpy()
        local_concs = scaled_concs[mask].to_numpy()
        print(local_times)
        print(local_concs)

        # Require at least 2 points to compute finite differences
        if len(local_times) < 2:
            print("[Warning] Not enough points to compute flux bounds, using global std.")
            # Fallback: global std of spline derivative
            full_t = np.linspace(corrected_times.min(), corrected_times.max(), 1000)
            full_derivs = spline_derivative(full_t)
            global_std = np.std(full_derivs, ddof=1)  # Use ddof=1 for sample std

            # Flip the sign of the flux?
            flux_est = -flux_est
            margin = max(z * global_std, min_flux_ci)
            lower = flux_est - margin
            upper = flux_est + margin
            return lower, upper

        dt = np.diff(local_times)
        dc = np.diff(local_concs)
        finite_derivs = dc / dt

        # Midpoints of finite differences
        t_mid = (local_times[:-1] + local_times[1:]) / 2

        # Residuals between spline and finite diff
        spline_vals_at_mid = spline_derivative(t_mid)
        residuals = finite_derivs - spline_vals_at_mid

        # Weights centered around t0
        weights = gaussian_weights(t_mid, t, sigma=flux_bounds_sigma)
        weights /= weights.sum()

        # ddof=1 is only used for sample counts, and not for weights
        std = weighted_std(residuals, weights)
        # std = np.std(residuals, ddof=1) if len(residuals) > 1 else 0.0

        # NOTE: Flip the sign of the flux?
        flux_est = -flux_est
        margin = max(z * std, min_flux_ci)
        lower = flux_est - margin
        upper = flux_est + margin

        print(f"[t={t:.2f}]: std={std:.4f}, flux={flux_est:.4f}, lb={lower:.4f}, ub={upper:.4f}")

        return (lower, upper)

    # Optional plotting
    if plot:
        exp_name = os.path.basename(csv_path).removesuffix("_areas.csv")
        plot_adjusted_flux(corrected_times, scaled_concs, flux_fn, spline,
                           exp_name = exp_name, target_col = target_col)

    return flux_fn

# 1. Load model
objective = "ATP_sink"
modelfile = "/data/local/jy1008/MA-host-microbiome/nmr-cdiff/data/icdf843.json"
model = cb.io.load_json_model(modelfile)
model.objective = objective

# for csv_path in pro_csv_paths:
# csv_path = pro_csv_paths[0]
pro_csv_paths = [
    "concentration_estimation/Data1_13CPro1_areas.csv",
    "concentration_estimation/Data2_13CPro2_areas.csv",
    "concentration_estimation/Data3_13CPro3_areas.csv"
]

# Start time calculation using isocaproate
for csv_path in pro_csv_paths:
    start_time = get_time_correction(csv_path, thresh=0.05, plot=True)
    print(f"Start time for {os.path.basename(csv_path)}: {start_time:.2f} hours")

for csv_path in pro_csv_paths:

target_col = "Proline 4.2469"
initial_concentration = 15.0  # mMol
pro_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=4.0, 
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=True)
target_col = "Glucose 5.2254"
initial_concentration = 27.77777778  # mMol
# 4.0 for Pro1, 16.0 for Pro2, 4.0 for Pro3
glc_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=4.0, 
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=True)
target_col = "Valine 1.0253"
initial_concentration = 2.564102564  # mMol
# 4.0 for Pro1, 0.1 for Pro2, 0.1 for Pro3
val_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=0.1, 
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=True)
target_col = "Leucine 0.9493"
initial_concentration = 7.633587786  # mMol
leu_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=0.2,
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=True)
# 0.2 for Pro1, 1.0 for Pro2, 0.05 for Pro3
target_col = "Isoluecine 0.9258"
initial_concentration = 2.290076336  # mMol
ile_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=0.05,
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=True)


constraints = {
    "Ex_proL": MetaboliteConstraint("Ex_proL", pro_flux_fn),
    "Ex_glc": MetaboliteConstraint("Ex_glc", glc_flux_fn),
    "Ex_valL": MetaboliteConstraint("Ex_valL", val_flux_fn),
    "Ex_leuL": MetaboliteConstraint("Ex_leuL", leu_flux_fn),
    "Ex_ileL": MetaboliteConstraint("Ex_ileL", ile_flux_fn)
}

# 3. Run dFBA
# ID_135: proL_c --> proD_c (proline racemase)
# ID_314: proline --> 5-aminovalerate

# Pro1 time_range = (0, 48), steps_per_hour=5
# Pro2 time_range = (-7, 33), steps_per_hour=5
# Pro3 time_range = (-11, 30), steps_per_hour=5
sim = dFBA(
    model=model,
    objective=objective,
    constraints=constraints,
    # time_range=(0, 48),
    # time_range=(-7, 33),
    time_range=(-11, 30),
    steps_per_hour=5,
    # tracked_reactions=["ATP_sink", "ID_314", "ID_135"],
    # tracked_reactions=["ATP_sink", "ID_314", "ID_135", 
    #                    "Trans_glc", "Ex_proL", "Ex_leuL", 
    #                    "Ex_valL", "Ex_ileL", "Ex_thrL", "Sec_ac", 
    #                    "Ex_alaL", "Ex_cysL", "Sec_ppa", "Sec_2abut", "ID_326"],
    tracked_reactions=["ATP_sink", "ID_251", "ID_252", "ID_512", "ID_474", "ID_49",
                    "ID_280", "ID_321", "ID_146", "ID_366", "ID_1021311", "ID_1021312",
                    "ID_1021313", "PPAKr", "ATPsynth4_1", "BUK", "IACK", "ImzACK", "HPhACK",
                    "Ex_proL", "Ex_glc", "Ex_valL", "Ex_leuL", "Ex_ileL"],
    fva=True
)

sim.run()
sim.export_results()

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
reactions = ["ATP_sink", "ID_252", "ID_280", "ID_321", "ID_146", "ID_366", "ATPsynth4_1", "ImzACK",
             "Ex_proL", "Ex_glc", "Ex_valL", "Ex_leuL", "Ex_ileL"]
df_conc = plot_integrated_fluxes(df, reactions, initial_conc=0)
plot_raw_fluxes(df, reactions)

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


def plot_raw_fluxes_html(flux_df, reactions, outname="raw_fluxes.html"):
    """
    Plot raw fluxes for specified reactions and save as HTML.
    """
    fig = go.Figure()

    for rxn in reactions:
        if rxn in flux_df.columns:
            fig.add_trace(go.Scatter(
                x=flux_df.index,
                y=flux_df[rxn],
                mode='lines',
                name=rxn,
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

plot_raw_fluxes_html(flux_df, interesting_reactions, outname="interesting_fluxes_all_pro1.html")
plot_raw_fluxes_html(flux_df, bounding_reactions, outname="interesting_fluxes_v2_bounding_pro1.html")
plot_raw_fluxes_html(flux_df, interesting_reactions_gt2, outname="interesting_fluxes_v2_gt2_pro1.html")
plot_raw_fluxes_html(flux_df, interesting_reactions_lt2, outname="interesting_fluxes_v2_lt2_pro1.html")

'''