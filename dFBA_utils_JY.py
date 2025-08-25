import numpy as np
import cobra as cb
import networkx as nx
from dFBA_JY import dFBA
from networkx.drawing.nx_agraph import graphviz_layout
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from scipy import integrate

def print_rxns(model, metab, role="all"):
    rxn_list = []
    # match string to metabolites
    met_list = [met.id for met in model.metabolites if metab in met.id]
    for met in met_list:
        print(f"----- {met} -----")
        temp = model.metabolites.get_by_id(met)
        for rxn in temp.reactions:
            if role == "reactant" and temp not in rxn.reactants:
                continue
            if role == "product" and temp not in rxn.products:
                continue
            print(rxn.id, rxn.reaction)
            rxn_list.append(rxn)
    return rxn_list


def generate_rxn_graph(model, metab, role="all"):
    G = nx.DiGraph()

    # rxn_list = print_rxns("proL")
    # rxn_list = print_rxns("5apn_c") + print_rxns("proD_c")
    # rxn_list = print_rxns("nadh", role="product")
    rxn_list = print_rxns(model, metab, role=role)

    # Iterate through all reactions in the model
    # for rxn in model.reactions:
    for rxn in rxn_list:
        for met in rxn.reactants:
            for prod in rxn.products:
                # add an edge from reactant to product via reaction
                G.add_edge(met.id, prod.id, reaction=rxn.id)

    # subgraph = G.subgraph(sub_nodes)
    # pos = nx.spring_layout(G, k=0.5, iterations=100)  # try adjusting k
    # nx.draw(G, pos, with_labels=True)

    pos = graphviz_layout(G, prog="dot")
    edge_labels = nx.get_edge_attributes(G, 'reaction')
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=800, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.tight_layout()
    plt.show()

    return G

# For use in estimating the bounds of the fluxes from sample data
# Gaussian weight function
def gaussian_weights(t_points, t0, sigma):
    return np.exp(-0.5 * ((t_points - t0) / sigma) ** 2)

# Compute weighted std
def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    return np.sqrt(variance)

def simulate_proline(seed=None):
    """
    Create a descending logistic function to simulate proline production.
    """
    timecourse = np.linspace(0, 48, 100)

    # Parameters for the logistic function
    L = 5.22  # Maximum proline concentration
    k = 1.0   # Growth rate
    x0 = 24.0   # Midpoint of the curve (when t=24, proline is at half max)

    # Calculate proline concentration over time
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=0.25, size=len(timecourse))
    proline_concentration = L * (1 - 1 / (1 + np.exp(-k * (timecourse - x0))) ) + noise
 
    return timecourse, proline_concentration

def simulate_glucose(seed=None):
    """
    Create a descending logistic function to simulate glucose production.
    """
    timecourse = np.linspace(0, 48, 100)

    # Parameters for the logistic function
    L = 20  # Maximum proline concentration
    k = 0.15   # Growth rate
    x0 = 36   # Midpoint of the curve (when t=24, proline is at half max)

    # Calculate proline concentration over time
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(loc=0.0, scale=0.25, size=len(timecourse))
    concentration = 10 + L * (1 - 1 / (1 + np.exp(-k * (timecourse - x0))) ) + noise
 
    return timecourse, concentration


# Logistic function
# def logistic(t, A, K, B, M):
#     return A + (K - A) / (1 + np.exp(-B * (t - M)))
def logistic(t, A, K, B, M):
    x = B * (t - M)
    # For large negative x, exp(x) ~ 0, for large positive x, exp(-x) ~ 0
    # Use np.where to avoid overflow
    result = np.where(
        x >= 0,
        A + (K - A) / (1 + np.exp(-x)),
        A + (K - A) * np.exp(x) / (1 + np.exp(x))
    )
    return result

def logistic_derivative(t, A, K, B, M):
    exp_term = np.exp(-B * (t - M))
    numerator = (K - A) * B * exp_term
    denominator = (1 + exp_term) ** 2
    return numerator / denominator  # dC/dt

def logistic_fit(times, concentrations):
    from scipy.optimize import curve_fit
    A0 = np.min(concentrations)
    K0 = np.max(concentrations)
    M0 = np.median(times)
    
    # Set initial slope B0 negative if start > end, else positive
    B0 = -0.1 if concentrations[0] > concentrations[-1] else 0.1

    p0 = [A0, K0, B0, M0]
    
    # Optional: allow B to be negative in bounds
    bounds = ([0, 0, -np.inf, min(times)], [np.inf, np.inf, np.inf, max(times)])

    params, covariance = curve_fit(
        logistic, times, concentrations, p0=p0,
        bounds=bounds,
        maxfev=10000
    )
    return params


def plot_integrated_fluxes(df, reactions, initial_conc=0):
    """
    Integrate fluxes over time to get concentrations and plot them.
    
    Parameters:
    - df: pandas DataFrame with time as index and flux columns
    - reactions: list of reaction IDs (column names) to integrate and plot
    - initial_conc: initial concentration value for all reactions (default 0)
    """
    df = df.sort_index()  # Ensure sorted by time
    time = df.index.values
    
    plt.figure(figsize=(8,6))
    
    for rxn in reactions:
        if rxn not in df.columns:
            print(f"Warning: reaction '{rxn}' not found in DataFrame columns.")
            continue
        
        flux = df[rxn].values
        concentration = integrate.cumulative_trapezoid(flux, time, initial=0) + initial_conc
        
        print(rxn)
        print(flux)

        # Add new column for integrated concentration
        conc_col = f"{rxn}_conc"
        df[conc_col] = concentration
        
        plt.plot(time, concentration, label=f"{rxn} concentration")
    
    plt.xlabel("Time")
    plt.ylabel("Concentration (integrated flux)")
    plt.title("Integrated Concentrations from Fluxes")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return df  # optionally return the updated DataFrame with new concentration columns



def plot_raw_fluxes(df, reactions, outname="dfba_flux", model=None, plot_bounds=False):
    """
    Plot fluxes with optional min/max shading.
    
    Parameters:
    - df: pandas DataFrame with time as index and flux columns
    - reactions: list of reaction IDs (column names) to plot
    """
    df = df.sort_index()  # Ensure sorted by time
    time = df.index.values
    
    plt.figure(figsize=(12, 6))
    
    for rxn in reactions:
        if rxn not in df.columns:
            print(f"Warning: reaction '{rxn}' not found in DataFrame columns.")
            continue
        
        flux = df[rxn].values
        rxn_name = ""
        if model is not None:
            rxn_obj = model.reactions.get_by_id(rxn)
            rxn_name = rxn_obj.name
        line, = plt.plot(time, flux, label=f"{rxn_name} ({rxn})")  # line handle to get color
        
        # Check if min/max columns exist
        min_col = f"{rxn}_min"
        max_col = f"{rxn}_max"
        
        if plot_bounds and min_col in df.columns and max_col in df.columns:
            plt.fill_between(
                time,
                df[min_col].values,
                df[max_col].values,
                color=line.get_color(),   # match line color
                alpha=0.3
            )
    
    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.title("Raw fluxes")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.75))
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(right=0.5)
    # plt.show()
    plt.savefig(f"{outname}.pdf", dpi=300, bbox_inches="tight")