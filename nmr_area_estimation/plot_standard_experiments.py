import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data_Standards_13C"
# input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data8_13CGlc2"
# input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data9_13CGlc3"

# input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data9_13CGlc3"

exp_name = "traces_13C_annot"

ref_concs = pd.read_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/20220325_13CGlc_Standards/combined_concentrations.csv")
ref_concs.columns = ["Time"] + ["13C_" + col + "_mMol" for col in ref_concs.columns[1:]]

records = []

for fname in os.listdir(input_dir):
    if fname.endswith(".json"):
        with open(os.path.join(input_dir, fname)) as f:
            data = json.load(f)
            if isinstance(data, dict):
                records.append(data)
            else:
                raise ValueError(f"{fname} is not a flat dict JSON")

# Convert list of dicts → DataFrame
df = pd.DataFrame(records)
df["total_area"] = df["total_area"].clip(lower=0)

df = df[df["experiment_name"] == exp_name]

df_grouped = df.pivot(index="metabolite", columns="time", values="total_area")
df_grouped = df_grouped.T
df_grouped = df_grouped.reset_index().rename(columns={"time": "Time"})

df_grouped["13C_Glucose"] = df_grouped["13C_Glucose"] / 4.0
df_grouped["13C_Acetate"] = df_grouped["13C_Acetate"] / 2.0
df_grouped["13C_Butyrate"] = df_grouped["13C_Butyrate"] / 2.0
df_grouped["13C_Ethanol"] = df_grouped["13C_Ethanol"] / 2.0
df_grouped["13C_Alanine"] = df_grouped["13C_Alanine"] / 2.0

merged = pd.merge(df_grouped, ref_concs, on='Time')

metab_list = ["13C_Glucose", "13C_Acetate", "13C_Alanine", "13C_Butyrate", "13C_Ethanol"]
for metab in metab_list:
    merged[f"{metab}_ratio"] = merged[metab] / merged[f"{metab}_mMol"]


def plot_rel(x, y, xlabel, ylabel):
    # Convert to NumPy arrays in case they aren't
    x = np.array(x)
    y = np.array(y)
    
    # Remove NaN and inf values
    mask = np.isfinite(x) & np.isfinite(y)  # True for values that are not NaN or inf
    x_clean = x[mask]
    y_clean = y[mask]

    # Compute linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    y_fit = slope * x_clean + intercept

    # Plot
    plt.scatter(x_clean, y_clean)
    plt.plot(x_clean, y_fit, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    # Add trendline equation and R^2
    plt.text(
        0.05, 0.95,
        f"y={slope:.3f}x+{intercept:.3f}\\n$R^2$={r_value**2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top'
    )
    
    plt.show()

def plot_rel_color(x, y, xlabel, ylabel, c=None, cmap="viridis"):
    # Convert to NumPy arrays in case they aren't
    x = np.array(x)
    y = np.array(y)
    if c is not None:
        c = np.array(c)

    # Remove NaN and inf values (apply mask to c as well if given)
    mask = np.isfinite(x) & np.isfinite(y)
    if c is not None:
        mask = mask & np.isfinite(c)

    x_clean = x[mask]
    y_clean = y[mask]
    c_clean = c[mask] if c is not None else None

    # Compute linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    y_fit = slope * x_clean + intercept

    # Plot with optional coloring
    sc = plt.scatter(x_clean, y_clean, c=c_clean, cmap=cmap)
    plt.plot(x_clean, y_fit, color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Add colorbar if c was provided
    if c is not None:
        plt.colorbar(sc, label="Magnitude")

    # Add trendline equation and R^2
    plt.text(
        0.05, 0.95,
        f"y={slope:.3f}x+{intercept:.3f}\\n$R^2$={r_value**2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top'
    )

    plt.show()






plot_rel(merged["13C_Glucose_ratio"], merged["13C_Acetate_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Acetate]/Peak Area mMol/a.u.")

plot_rel(merged["13C_Glucose_ratio"], merged["13C_Alanine_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Alanine]/Peak Area mMol/a.u.")

plot_rel(merged["13C_Glucose_ratio"], merged["13C_Butyrate_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Butyrate]/Peak Area mMol/a.u.")

plot_rel(merged["13C_Glucose_ratio"], merged["13C_Ethanol_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Ethanol]/Peak Area mMol/a.u.")



# same procedure for 1H data


input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data_Standards_1H"

exp_name = "traces_1H_annot"

ref_concs = pd.read_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/20220325_13CGlc_Standards/combined_concentrations.csv")
ref_concs.columns = ["Time"] + ["13C_" + col + "_mMol" for col in ref_concs.columns[1:]]

records = []

for fname in os.listdir(input_dir):
    if fname.endswith(".json"):
        with open(os.path.join(input_dir, fname)) as f:
            data = json.load(f)
            if isinstance(data, dict):
                records.append(data)
            else:
                raise ValueError(f"{fname} is not a flat dict JSON")

# Convert list of dicts → DataFrame
df = pd.DataFrame(records)
df["total_area"] = df["total_area"].clip(lower=0)

df = df[df["experiment_name"] == exp_name]

# Apply a filter to the total area? In Ext. Fig 12, a filter of 18 is used
# df[df["total_area"] > 20]

df_grouped = df.pivot(index="metabolite", columns="time", values="total_area")
df_grouped = df_grouped.T
df_grouped = df_grouped.reset_index().rename(columns={"time": "Time"})

# df_grouped["13C_Glucose"] = df_grouped["13C_Glucose"] / 4.0
# df_grouped["13C_Acetate"] = df_grouped["13C_Acetate"] / 2.0
# df_grouped["13C_Butyrate"] = df_grouped["13C_Butyrate"] / 2.0
# df_grouped["13C_Ethanol"] = df_grouped["13C_Ethanol"] / 2.0
# df_grouped["13C_Alanine"] = df_grouped["13C_Alanine"] / 2.0

merged = pd.merge(df_grouped, ref_concs, on='Time')

metab_list = ["13C_Glucose", "13C_Acetate", "13C_Alanine", "13C_Butyrate", "13C_Ethanol"]
for metab in metab_list:
    merged[f"{metab}_ratio"] = merged[metab] / merged[f"{metab}_mMol"]


plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Acetate_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Acetate]/Peak Area mMol/a.u.",
                c=merged["13C_Acetate"])

plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Alanine_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Alanine]/Peak Area mMol/a.u.",
                c=merged["13C_Alanine"])

plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Butyrate_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Butyrate]/Peak Area mMol/a.u.",
                c=merged["13C_Butyrate"])

plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Ethanol_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Ethanol]/Peak Area mMol/a.u.",
                c=merged["13C_Ethanol"])