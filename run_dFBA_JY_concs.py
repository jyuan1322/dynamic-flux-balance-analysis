import os
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

from dFBA_JY import dFBA, MetaboliteConstraint
from dFBA_utils_JY import *

# 1. Load model
objective = "ATP_sink"
modelfile = "/data/local/jy1008/MA-host-microbiome/nmr-cdiff/data/icdf843.json"
model = cb.io.load_json_model(modelfile)
model.objective = objective



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
# for csv_path in pro_csv_paths:
#     start_time = get_time_correction(csv_path, thresh=0.05, plot=True)
#     print(f"Start time for {os.path.basename(csv_path)}: {start_time:.2f} hours")

# Create a function f(t) which returns a lower and upper bound for the flux at time t.
# This version calculates bounds based on a mean and std obtained directly from the
# sample data.
def flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=0.5,
                              flux_bounds_window = 5, flux_bounds_sigma = 1, plot=False):
    df = pd.read_csv(csv_path)
    # Correct for time offset
    start_time = get_time_correction(csv_path, thresh=0.05, plot=plot)
    corrected_times = df['Time'] - start_time

    # Scale the concentrations to mMol using the recorded initial concentration
    # Use these values going forward
    scale_factor = initial_concentration / df[target_col][0]
    scaled_concs = df[target_col] * scale_factor
    # Subtract minimum to normalize to 0 if there are negative values
    if(scaled_concs.min() < 0):
        scaled_concs = scaled_concs - scaled_concs.min()

    # To estimate the mean flux, precompute the interpolation spline and its derivative
    spline = UnivariateSpline(corrected_times, scaled_concs, s=smoothing_factor)
    spline_derivative = spline.derivative()
    # To estimate the CI of the flux, calculate point-wise derivatives in a window around t
    ci = 0.95  # Confidence interval
    z = norm.ppf((1 + ci) / 2)

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

            mean_flux = spline_derivative(t)
            # lower = mean_flux - z * global_std
            # upper = mean_flux + z * global_std

            # Flip the sign of the flux?
            mean_flux = -mean_flux
            margin = max(z * global_std, min_flux_ci)
            lower = mean_flux - margin
            upper = mean_flux + margin
            return mean_flux, lower, upper

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

        # margin = max(z * std, 1.0)
        # margin = z * std
        # margin = z * std # scale factor for wider bounds
        # lower = flux_est - margin
        # upper = max(flux_est + margin, 0)
        # upper = flux_est + margin

        # NOTE: Flip the sign of the flux?
        flux_est = -flux_est
        margin = max(z * std, min_flux_ci)
        lower = flux_est - margin
        upper = flux_est + margin

        print(f"[t={t:.2f}]: std={std:.4f}, flux={flux_est:.4f}, lb={lower:.4f}, ub={upper:.4f}")

        return (flux_est, lower, upper)

    def flux_fn_wrap(t):
        """
        Returns only the lower and upper bounds for the flux at time t.
        """
        mean, lower, upper = flux_fn(t)
        return (lower, upper)

    # Optional plotting
    if plot:
        t_fit = np.linspace(corrected_times.min(), corrected_times.max(), 1000)
        # Evaluate flux_fn at each t
        bounds = [flux_fn(t) for t in t_fit]

        # Unpack into lower and upper bound arrays
        means, lower_bounds, upper_bounds = zip(*bounds)
        means = np.array(means)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)

        plt.figure(figsize=(8, 5))

        # Flux CI bounds
        plt.fill_between(t_fit, lower_bounds, upper_bounds, color='red', alpha=0.3, label='Flux ± 0.95CI')

        # Mean Flux
        plt.plot(t_fit, means, label='Mean Flux', color='red')

        # Experimental data
        plt.scatter(corrected_times, scaled_concs, label="Experimental Concentrations", s=16, color='black')

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
        plt.plot(corrected_times, finite_deriv, 'o', label='Finite Difference dC/dt', markersize=4, color='blue')
        
        plt.xlabel('Time')
        plt.ylabel(f"{target_col} (scaled to {initial_concentration} mMol)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return flux_fn_wrap


# for csv_path in pro_csv_paths:
# csv_path = pro_csv_paths[0]
pro_csv_paths = [
    "concentration_estimation/Data1_13CPro1_areas.csv",
    "concentration_estimation/Data2_13CPro2_areas.csv",
    "concentration_estimation/Data3_13CPro3_areas.csv"
]
for csv_path in pro_csv_paths:

target_col = "Proline 4.2469"
initial_concentration = 15.0  # mMol
pro_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=4.0, 
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=False)

target_col = "Glucose 5.2254"
initial_concentration = 27.77777778  # mMol
glc_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=4.0, 
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=False)
target_col = "Valine 1.0253"
initial_concentration = 2.564102564  # mMol
val_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=4.0, 
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=False)
target_col = "Leucine 0.9493"
initial_concentration = 7.633587786  # mMol
leu_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=4.0, 
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=False)
target_col = "Isoluecine 0.9258"
initial_concentration = 2.290076336  # mMol
ile_flux_fn = flux_function_with_bounds(target_col, csv_path, initial_concentration, smoothing_factor=4.0, 
                                        flux_bounds_window=5, flux_bounds_sigma=1, plot=False)


t = np.linspace(0, 48, 100)
glc_flux_bounds = [glc_flux_fn(ti) for ti in t]
glc_flux_bounds = np.array(glc_flux_bounds)
plt.plot(t, glc_flux_bounds)
plt.show()

# print(sim.model.reactions.get_by_id("Ex_proL").lower_bound)
# print(sim.model.reactions.get_by_id("Ex_proL").upper_bound)



# Ask about sign issue: 
# Ex_proL bounds of [-1000, 0] don't work, but [0, 1000] does.
# Is it from the perspective of the medium vs cells?
# A bound of [0, 1.2] works, but [0, 1.1] doesn't.
def dummy_proline_constraint(t):
    lb = 0
    ub = 1.1
    return (lb, ub)

def dummy_glucose_constraint(t):
    lb = 0
    ub = 2
    return(lb, ub)

constraints = {
    # "Ex_proL": MetaboliteConstraint("Ex_proL", dummy_proline_constraint)# ,
    # "Ex_proL": MetaboliteConstraint("Ex_proL", pro_flux_fn),
    "Ex_glc": MetaboliteConstraint("Ex_glc", glc_flux_fn)# 
    # "Ex_glc": MetaboliteConstraint("Ex_glc", dummy_glucose_constraint)
    # "Ex_valL": MetaboliteConstraint("Ex_valL", val_flux_fn),
    # "Ex_leuL": MetaboliteConstraint("Ex_leuL", leu_flux_fn),
    # "Ex_ileL": MetaboliteConstraint("Ex_ileL", ile_flux_fn)
}

# 3. Run dFBA
# ID_135: proL_c --> proD_c (proline racemase)
# ID_314: proline --> 5-aminovalerate
sim = dFBA(
    model=model,
    objective=objective,
    constraints=constraints,
    time_range=(0, 48),
    steps_per_hour=2,
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

plot_raw_fluxes_html(flux_df, interesting_reactions, outname="interesting_fluxes_all.html")
plot_raw_fluxes_html(flux_df, bounding_reactions, outname="interesting_fluxes_v2_bounding.html")
plot_raw_fluxes_html(flux_df, interesting_reactions_gt2, outname="interesting_fluxes_v2_gt2.html")
plot_raw_fluxes_html(flux_df, interesting_reactions_lt2, outname="interesting_fluxes_v2_lt2.html")

