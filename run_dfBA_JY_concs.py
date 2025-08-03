import cobra as cb
import networkx as nx
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.drawing.nx_agraph import graphviz_layout
from cycler import cycler
import plotly.express as px
import plotly.graph_objects as go

from dFBA_JY import dFBA, MetaboliteConstraint
from dfba_utils_JY import *

# 1. Load model
objective = "ATP_sink"
modelfile = "/data/local/jy1008/MA-host-microbiome/nmr-cdiff/data/icdf843.json"
model = cb.io.load_json_model(modelfile)
model.objective = objective


def fit_logistic_with_bounds(target_col, csv_paths, initial_concentration, constraint_func=None, plot=False):
    dfs = [pd.read_csv(path) for path in csv_paths]

    # Subtract minimum to normalize to 0
    for df in dfs:
        df[target_col] = df[target_col] - df[target_col].min()

    # Fit individual curves
    params_list = []
    for df in dfs:
        times = df["Time"].values
        values = df[target_col].values
        params = logistic_fit(times, values)
        params_list.append(params)

    # Shared time grid
    min_time = min(df['Time'].min() for df in dfs)
    max_time = max(df['Time'].max() for df in dfs)
    t_fit = np.linspace(min_time, max_time, 200)

    # Evaluate curves
    curves = [logistic(t_fit, *params) for params in params_list]
    curve_mean = np.mean(curves, axis=0)
    curve_se = np.std(curves, axis=0, ddof=1) / np.sqrt(len(curves))

    # ---- Rescale to match specified initial concentration ----
    scale_factor = initial_concentration / curve_mean[0]
    curve_mean *= scale_factor
    curve_upper = curve_mean + curve_se * scale_factor
    curve_lower = curve_mean - curve_se * scale_factor

    # Rescale data in each DataFrame for plotting
    for df in dfs:
        df[target_col] *= scale_factor

    # Re-fit logistic functions to rescaled bounds
    params_lb = logistic_fit(t_fit, curve_lower)
    params_ub = logistic_fit(t_fit, curve_upper)

    if constraint_func is not None:
        constraint = constraint_func(params_lb, params_ub)
    else:
        constraint = None

    # Optional plotting
    if plot:
        # Compute smooth logistic curves from bounding fits
        lower_fit = logistic(t_fit, *params_lb)
        upper_fit = logistic(t_fit, *params_ub)

        plt.figure(figsize=(8, 5))

        # Raw SE bounds
        plt.fill_between(t_fit, curve_lower, curve_upper, color='gray', alpha=0.3, label='Pointwise ±SE')

        # Logistic bounding fits
        plt.fill_between(t_fit, lower_fit, upper_fit, color='skyblue', alpha=0.4, label='Logistic bounds')

        # Mean logistic
        plt.plot(t_fit, curve_mean, 'b-', label='Mean logistic fit')

        # Experimental data
        markers = ['o', '^', 's', 'D', 'v', 'x', '*', 'P', 'h', 'H']
        colors = plt.cm.tab10.colors

        for i, df in enumerate(dfs):
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.scatter(df['Time'], df[target_col], marker=marker, color=color, label=f'Experiment {i+1}')

        plt.xlabel('Time')
        plt.ylabel(f"{target_col} (scaled to {initial_concentration})")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return params_lb, params_ub, constraint


# need to define timecourse somewhere
timecourse, proL_concs = simulate_proline(seed=42)
proL_params = logistic_fit(timecourse, proL_concs)

timecourse, glu_concs = simulate_glucose(seed=42)
glu_params = logistic_fit(timecourse, glu_concs)


# Generate fitted curve
def plot_fit(timecourse, concs, params, outname):
    # t_fit = np.linspace(times.min(), times.max(), 100)
    y_fit = logistic(timecourse, *params)
    y_flux = logistic_derivative(timecourse, *params)

    # Plot data + fit
    # plt.scatter(timecourse, proL_concs, label="simul", color="blue")
    plt.scatter(timecourse, concs, label="simul", color="blue")
    plt.plot(timecourse, y_fit, label="Logistic fit", color="red")
    plt.plot(timecourse, y_flux, label="Logistic flux", color="green")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    # plt.savefig("proline_simulated_fit.pdf", dpi=300)
    # plt.savefig("glucose_simulated_fit.pdf", dpi=300)
    plt.savefig(outname, dpi=300)
    plt.close()
plot_fit(timecourse, proL_concs, proL_params, "proline_simulated_fit.pdf")
plot_fit(timecourse, glu_concs, glu_params, "glucose_simulated_fit.pdf")


# create constraint functions dynamically based on trained parameters
def make_logistic_flux_fn(*params, dummy_stderr=0.05):
    # A_fit, K_fit, B_fit, M_fit = params
    def logistic_flux_fn(t):
        # 'a' and 'b' treated as constants here
        val = logistic_derivative(t, *params)
        # multiply by -1 because positive flux is when proL is consumed
        val *= -1
        print(f"[DEBUG] t={t:.1f}, dC/dt = {val:.4e}")  # <- Add this
        # upper and lower bounds are the same
        # min(x, x) reduces the chance of numerical errors
        # return (min(val, val), max(val, val))
        lb = val * (1 - dummy_stderr)
        ub = val * (1 + dummy_stderr)
        return (lb, ub)
    return logistic_flux_fn

def make_logistic_flux_fn_bounded(params_lb, params_ub):
    # A_fit, K_fit, B_fit, M_fit = params
    def logistic_flux_fn(t):
        # 'a' and 'b' treated as constants here
        # val = logistic_derivative(t, *params)
        val_lb = logistic_derivative(t, *params_lb)
        val_ub = logistic_derivative(t, *params_ub)
        # multiply by -1 because positive flux is when proL is consumed
        val_lb *= -1
        val_ub *= -1
        print(f"[DEBUG] t={t:.1f}, dC/dt = [{val_lb:.4e}, ]{val_ub:.4e}")
        # upper and lower bounds are the same
        # min(x, x) reduces the chance of numerical errors
        # return (min(val, val), max(val, val))

        # Ensure lower bound is actually lower
        lb = min(val_lb, val_ub)
        ub = max(val_lb, val_ub)
        return (lb, ub)
    return logistic_flux_fn

# proL_constraint_func = make_logistic_flux_fn(*proL_params)
# glu_constraint_func = make_logistic_flux_fn(*glu_params)



pro_lb, pro_ub, pro_constraint_fn = fit_logistic_with_bounds(
    target_col="Proline 4.2469",
    csv_paths=[
        "concentration_estimation/Data1_13CPro1_areas.csv",
        "concentration_estimation/Data2_13CPro2_areas.csv",
        "concentration_estimation/Data3_13CPro3_areas.csv"
    ],
    initial_concentration=15.0,
    constraint_func=make_logistic_flux_fn_bounded,
    plot=False
)

glc_lb, glc_ub, glc_constraint_fn = fit_logistic_with_bounds(
    target_col="Glucose 5.2254",
    csv_paths=[
        "concentration_estimation/Data1_13CPro1_areas.csv",
        "concentration_estimation/Data2_13CPro2_areas.csv",
        "concentration_estimation/Data3_13CPro3_areas.csv"
    ],
    initial_concentration=27.77777778,
    constraint_func=make_logistic_flux_fn_bounded,
    plot=False
)

val_lb, val_ub, val_constraint_fn = fit_logistic_with_bounds(
    target_col="Valine 1.0253",
    csv_paths=[
        "concentration_estimation/Data1_13CPro1_areas.csv",
        "concentration_estimation/Data2_13CPro2_areas.csv",
        "concentration_estimation/Data3_13CPro3_areas.csv"
    ],
    initial_concentration=2.564102564,
    constraint_func=make_logistic_flux_fn_bounded,
    plot=False
)

leu_lb, leu_ub, leu_constraint_fn = fit_logistic_with_bounds(
    target_col="Leucine 0.9493",
    csv_paths=[
        "concentration_estimation/Data1_13CPro1_areas.csv",
        "concentration_estimation/Data2_13CPro2_areas.csv",
        "concentration_estimation/Data3_13CPro3_areas.csv"
    ],
    initial_concentration=7.633587786,
    constraint_func=make_logistic_flux_fn_bounded,
    plot=False
)

ile_lb, ile_ub, ile_constraint_fn = fit_logistic_with_bounds(
    target_col="Isoluecine 0.9258",
    csv_paths=[
        "concentration_estimation/Data1_13CPro1_areas.csv",
        "concentration_estimation/Data2_13CPro2_areas.csv",
        "concentration_estimation/Data3_13CPro3_areas.csv"
    ],
    initial_concentration=2.290076336,
    constraint_func=make_logistic_flux_fn_bounded,
    plot=False
)

constraints = {
    # "Ex_glc": MetaboliteConstraint("Ex_glc", glucose_constraint)
    # "Ex_proL": MetaboliteConstraint("Ex_proL", proline_constraint)
    # "Ex_proL": MetaboliteConstraint("Ex_proL", proL_constraint_func),
    # "Ex_glc": MetaboliteConstraint("Ex_glc", glu_constraint_func)
    "Ex_proL": MetaboliteConstraint("Ex_proL", pro_constraint_fn),
    "Ex_glc": MetaboliteConstraint("Ex_glc", glc_constraint_fn),
    "Ex_valL": MetaboliteConstraint("Ex_valL", val_constraint_fn),
    "Ex_leuL": MetaboliteConstraint("Ex_leuL", leu_constraint_fn),
    "Ex_ileL": MetaboliteConstraint("Ex_ileL", ile_constraint_fn)
}

# 3. Run dFBA
# ID_135: proL_c --> proD_c (proline racemase)
# ID_314: proline --> 5-aminovalerate
sim = dFBA(
    model=model,
    objective=objective,
    constraints=constraints,
    time_range=(0, 48),
    steps_per_hour=5,
    # tracked_reactions=["ATP_sink", "ID_314", "ID_135"],
    # tracked_reactions=["ATP_sink", "ID_314", "ID_135", 
    #                    "Trans_glc", "Ex_proL", "Ex_leuL", 
    #                    "Ex_valL", "Ex_ileL", "Ex_thrL", "Sec_ac", 
    #                    "Ex_alaL", "Ex_cysL", "Sec_ppa", "Sec_2abut", "ID_326"],
    tracked_reactions=["ATP_sink", "ID_251", "ID_252", "ID_512", "ID_474", "ID_49",
                       "ID_280", "ID_321", "ID_146", "ID_366", "ID_1021311", "ID_1021312",
                       "ID_1021313", "PPAKr", "ATPsynth4_1", "BUK", "IACK", "ImzACK", "HPhACK"],
    fva=True
)
# Trans_glc: Phosphotransferase system (PTS), which imports glucose and phosphorylates it using PEP. ✅ Critical step in anaerobes like C. difficile.
# ID_657: Alternative to PTS—ATP-dependent phosphorylation of cytoplasmic glucose. Possibly used when PTS isn't.

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
reactions = ["ATP_sink", "ID_252", "ID_280", "ID_321", "ID_146", "ID_366", "ATPsynth4_1", "ImzACK"]
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

