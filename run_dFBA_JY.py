import cobra as cb
import networkx as nx
import numpy as np
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from networkx.drawing.nx_agraph import graphviz_layout
from cycler import cycler

from dFBA_JY import dFBA, MetaboliteConstraint
# from dfba_utils_JY import simulate_proline, logistic, logistic_fit
from dfba_utils_JY import *

# 1. Load model
objective = "ATP_sink"
modelfile = "/data/local/jy1008/MA-host-microbiome/nmr-cdiff/data/icdf843.json"
model = cb.io.load_json_model(modelfile)
model.objective = objective

# dfba.py line 90:
# the flux is reversed
"""
    # Calculate logistic solutions with flexible number of parameters
    signal, serr, exch, exerr = params.get_sol(t)
    exch_l, exch_u, signal_l, signal_u = params.get_bounds(t)
    # Reverse direction and set bounds for secretion reactions
    if met in ['glc', 'proL', 'leuL', 'valL', 'ileL', 'thrL']:
        rid = 'Ex_' + met
        exch, exch_l, exch_u = reverse_flux(exch, exch_l, exch_u)
"""
# 5.22 is the default upper bound
# model.reactions.get_by_id("Ex_proL").bounds = (0.0, 5.22)

# block uptake of other carbon sources
# for rxn in model.exchanges:
#     if rxn.id != "Ex_proL":
#         rxn.lower_bound = 0.0

# allowed = ["Ex_proL", "Ex_h", "Ex_nh4", "Ex_pi", "Ex_so4", "Ex_co2", "Ex_o2"]
# for rxn in model.exchanges:
#     rxn.lower_bound = -10.0 if rxn.id in allowed else 0.0







# 2. Define constraints
"""
def glucose_constraint(t):
    # linearly reduce glucose uptake over time
    lb = -10 + 0.2 * t
    ub = 0
    return (lb, ub)

def proline_constraint(t):
    # linearly reduce glucose uptake over time
    # lb = -10 + 0.2 * t
    # ub = -10 + 0.2 * t
    lb = 5.22 - 0.2 * t
    ub = 5.22 - 0.2 * t
    return (lb, ub)
"""

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

proL_constraint_func = make_logistic_flux_fn(*proL_params)
glu_constraint_func = make_logistic_flux_fn(*glu_params)

constraints = {
    # "Ex_glc": MetaboliteConstraint("Ex_glc", glucose_constraint)
    # "Ex_proL": MetaboliteConstraint("Ex_proL", proline_constraint)
    "Ex_proL": MetaboliteConstraint("Ex_proL", proL_constraint_func),
    "Ex_glc": MetaboliteConstraint("Ex_glc", glu_constraint_func)
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
reactions = ["ATP_sink", "ID_251", "ID_252", "ID_512", "ID_474", "ID_49",
             "ID_280", "ID_321", "ID_146", "ID_366", "ID_1021311", "ID_1021312",
             "ID_1021313", "PPAKr", "ATPsynth4_1", "BUK", "IACK", "ImzACK", "HPhACK"]
df_conc = plot_integrated_fluxes(df, reactions, initial_conc=0)
plot_raw_fluxes(df, reactions)



# Example: df with time as index or a column
# Ensure time is sorted ascending
df = df.sort_index()  # if time is index

time = df.index.values          # time points
flux = df["ID_314"].values      # flux (dC/dt)

# Initial concentration (can be zero or known)
# C0 = 0  

# Integrate flux over time
concentration = integrate.cumulative_trapezoid(flux, time, initial=0)

# Add as a new column to df
df["ID_314_conc"] = concentration

# Plot concentration over time
plt.plot(time, df["ID_314_conc"], label="ID_314 concentration")
plt.xlabel("Time")
plt.ylabel("Concentration (integrated)")
plt.legend()
plt.show()


# all nonzero fluxes, for diagnostics
for t in sim.timecourse:
    active = sim.all_fluxes[t][sim.all_fluxes[t] != 0.0]
    print(active.sort_values(key=abs, ascending=False))












def restrict_atp_producers_except(model, keep: list, metabs: list):
    """
    Restrict all ATP-producing reactions except those explicitly allowed.
    
    Parameters:
        model (cobra.Model): The COBRA model.
        keep (list of str): Reaction IDs to keep (not restrict).
        atp_id (str): The metabolite ID for ATP (default: "atp_c").
    """
    for metab in metabs:
        atp = model.metabolites.get_by_id(metab)
        blocked = []

        for rxn in atp.reactions:
            if atp in rxn.products and rxn.id not in keep:
                rxn.lower_bound = 0.0
                rxn.upper_bound = 0.0
                blocked.append(rxn.id)

    print(f"Blocked {len(blocked)} ATP-producing reactions:")
    for r in blocked:
        print(f"  - {r}")

allowed_reactions = ["ID_314", "ID_135"]
restrict_atp_producers_except(model, keep=allowed_reactions, metabs = ["atp_c", "pmf_c"])
# solution = model.optimize()
# print(solution.fluxes[["ATP_sink", "ID_314", "ID_135"]])




# Constraint seems to be applied to Ex_proL, Ex_leuL, etc.
# see line 194 in dfba.py
"""
    if met in ['glc', 'proL', 'leuL', 'valL', 'ileL', 'thrL']:
        rid = 'Ex_' + met
        exch, exch_l, exch_u = reverse_flux(exch, exch_l, exch_u)
        if met in ['leuL']:
            set_bounds(model, rid, update=update) # leave leucine unbounded
        else:
            set_bounds(model, rid, lower=exch_l, upper=exch_u, update=update)
"""

"""
proL_e
Ex_proL  <-- proL_e
Trans_proL atp_c + h2o_c + proL_e --> adp_c + pi_c + proL_c
Sec_proL proL_e -->
Trans_proL_PMF pmf_c + proL_e <=> proL_c


Trans_proL atp_c + h2o_c + proL_e --> adp_c + pi_c + proL_c
Sec_proL proL_e -->
Trans_proL_PMF pmf_c + proL_e <=> proL_c

ID_135 proL_c --> proD_c

ID_314 h_c + nadh_c + proD_c --> 5apn_c + nad_c + 1.1 pmf_c
"""

"""
# look up tracked reactions
leucine = model.metabolites.get_by_id("leuL_e")
for rxn in leucine.reactions:
    if leucine in rxn.reactants:
        print(rxn.id, rxn.reaction)
        
temp = model.metabolites.get_by_id("proL_e")
for rxn in temp.reactions:
    if temp in rxn.reactants:
        print(rxn.id, rxn.reaction)
"""




"""
>>> atp = model.metabolites.get_by_id("atp_c")
... for rxn in atp.reactions:
...     if atp in rxn.products:
...         print(f"ATP produced in {rxn.id}: {rxn.reaction}")
...
ATP produced in ID_474: adp_c + na_e + pi_c --> atp_c + h2o_c + na_c
ATP produced in IACK: adp_c + indap_c <=> atp_c + ind3ac_c
ATP produced in ID_49: adp_c + 4.0 h_e + pi_c --> atp_c + h2o_c + 4.0 h_c
ATP produced in ImzACK: adp_c + imzacp_c <=> atp_c + imzac_c
ATP produced in ATPsynth4_1: adp_c + pi_c + 4 pmf_c --> atp_c
ATP produced in ID_280: adp_c + ap_c <=> ac_c + atp_c
ATP produced in ID_512: adp_c + cbp_c --> atp_c + co2_c + nh3_c
ATP produced in HPhACK: adp_c + hphap_c <=> atp_c + hphac_c
ATP produced in BUK: adp_c + butp_c --> atp_c + but_c
ATP produced in ID_252: adp_c + pepyr_c <=> atp_c + pyr_c
ATP produced in ID_321: adp_c + isobutp_c --> atp_c + isobuta_c
ATP produced in ID_146: 2mbutp_c + adp_c --> 2mbut_c + atp_c
ATP produced in ID_251: adp_c + butp_c --> 2but_c + atp_c
ATP produced in ID_1021311: 12dhexdec3gp_c + adp_c <=> 12dgdihexdec_c + atp_c
ATP produced in ID_1021312: 12dtetradecg3p_c + adp_c <=> 12dgditetdec_c + atp_c
ATP produced in ID_366: adp_c + isop_c <=> atp_c + ival_c
ATP produced in ID_1021313: 12dioctdecg3p_c + adp_c <=> 12dgdioctdec_c + atp_c
ATP produced in PPAKr: adp_c + ppap_c <=> atp_c + ppa_c
"""

"""
in dfba_cfg.json,
Glucose has as a product 
Acetate, model_id: ac
internally, the code turns this into ac_c

"""



# evaluate a single time point
"""
def apply_constraints_at_time(model, constraints, t):
    for rxn_id, constraint in constraints.items():
        rxn = model.reactions.get_by_id(rxn_id)
        lb, ub = constraint.get_bounds(t)
        if ub < rxn.lower_bound:
            rxn.lower_bound = lb  # safe since lb <= ub
            rxn.upper_bound = ub
        else:
            rxn.upper_bound = ub
            rxn.lower_bound = lb

# Apply constraints for time t = 5.0 (example)
# t = 5.0
t = 50.0
apply_constraints_at_time(model, constraints, t)

# Optimize the model
solution = model.optimize()

# Inspect result
print("Objective value:", solution.objective_value)
print("Fluxes:")
print(solution.fluxes.get("ID_314", None))
print(solution.fluxes.get("ID_135", None))
print(solution.fluxes)
"""