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


# Add acquisition time rescale from Xi
acqus_times = {3:np.nan, 12:4096, 22:1024, 32:1024, 42:2048, 52:1024, 62:1024,
               72:1024, 82:1024, 92:4096, 102:1024, 112:4096, 122:1024}
acqus_times = pd.DataFrame(list(acqus_times.items()), columns=["Time", "Acqus_Time"])
merged = merged.merge(acqus_times, on="Time", how="left")
# remove entries without Acqus_Time
merged = merged[~merged["Acqus_Time"].isna()]

metab_list = ["13C_Glucose", "13C_Acetate", "13C_Alanine", "13C_Butyrate", "13C_Ethanol"]
for metab in metab_list:
    # rescale by Acqus_Time
    # merged[metab] = merged[metab] / merged["Acqus_Time"]
    # calculate ratio
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




merged.to_csv("standard_regression_plots_13C.csv", index=False)

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

# standard curve using just glucose
plot_rel(merged["13C_Glucose_mMol"], merged["13C_Glucose"],
                xlabel="[13C_Glucose] (mMol)",
                ylabel="13C_Glucose Peak Area (a.u.)")

plot_rel(merged["Acqus_Time"], merged["13C_Glucose"],
                xlabel="Number of scans",
                ylabel="13C_Glucose Peak Area (a.u.)")

plot_rel(merged["13C_Glucose_mMol"], merged["13C_Glucose"]/merged["Acqus_Time"],
                xlabel="[13C_Glucose] (mMol)",
                ylabel="13C_Glucose Peak Area (a.u.) / Number of scans")

plot_rel(merged["13C_Glucose_mMol"], merged["13C_Glucose"]/np.sqrt(merged["Acqus_Time"]),
                xlabel="[13C_Glucose] (mMol)",
                ylabel="13C_Glucose Peak Area (a.u.) / sqrt(Number of scans)")

# same procedure for 1H data


# input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data_Standards_1H"
input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data_Standards_1H_V3"

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

# Add acquisition time rescale from Xi
# acqus_times = {11: 0.5, 21:1, 31:1, 41:1/np.sqrt(2), 51:1, 61:1, 71:1,
#                81:1, 91:0.5, 101:1, 111:0.5, 121:1}
acqus_times = {11: 512, 21:128, 31:128, 41:256, 51:128, 61:128, 71:128,
               81:128, 91:512, 101:128, 111:512, 121:128}
acqus_times = pd.DataFrame(list(acqus_times.items()), columns=["Time", "Acqus_Time"])
merged = merged.merge(acqus_times, on="Time", how="left")
# remove entries without Acqus_Time
merged = merged[~merged["Acqus_Time"].isna()]

# Add scaling by total area

total_areas = {11: 1597985.85263905, 21:363026.03883776, 31:382053.42846874,
               41:636364.37455709, 51:298581.14083037, 61:348092.51816054,
               71:406638.68362965, 81:289010.53542787, 91:1409229.00448348,
               101:251404.54377862, 111:1317754.22866447, 121:215673.8013313}
total_areas = pd.DataFrame(list(total_areas.items()), columns=["Time", "Total_Area"])
merged = merged.merge(total_areas, on="Time", how="left")
# remove entries without Total_Area
merged = merged[~merged["Total_Area"].isna()]

metab_list = ["13C_Glucose", "13C_Acetate", "13C_Alanine", "13C_Butyrate", "13C_Ethanol"]
for metab in metab_list:
    # rescale by Acqus_Time
    # merged[metab] = merged[metab] / merged["Acqus_Time"]
    # calculate ratio
    merged[f"{metab}_ratio"] = merged[metab] / merged[f"{metab}_mMol"]

merged["13C_Alanine2_ratio"] = merged["13C_Alanine2"] / (merged["13C_Alanine_mMol"])

merged.to_csv("standard_regression_plots_1H.csv", index=False)

plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Acetate_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Acetate]/Peak Area mMol/a.u.",
                c=merged["13C_Acetate"])

plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Alanine_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Alanine]/Peak Area mMol/a.u.",
                c=merged["13C_Alanine"])

plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Alanine2_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Alanine2]/Peak Area mMol/a.u.",
                c=merged["13C_Alanine2"])

plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Butyrate_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Butyrate]/Peak Area mMol/a.u.",
                c=merged["13C_Butyrate"])

plot_rel_color(merged["13C_Glucose_ratio"], merged["13C_Ethanol_ratio"],
                xlabel="[13C_Glucose]/Peak Area mMol/a.u.",
                ylabel="[13C_Ethanol]/Peak Area mMol/a.u.",
                c=merged["13C_Ethanol"])


plot_rel_color(merged["13C_Glucose_mMol"], merged["13C_Glucose"],
                xlabel="13C_Glucose mMol",
                ylabel="13_C_Glucose Peak Area a.u.",
                c=merged["Time"])

plot_rel_color(merged["13C_Glucose_mMol"], merged["13C_Glucose"] / merged["Acqus_Time"],
                xlabel="13C_Glucose mMol",
                ylabel="13_C_Glucose Peak Area a.u. / Number of scans",
                c=merged["Time"])
plot_rel_color(merged["13C_Glucose_mMol"], merged["13C_Glucose"] / np.sqrt(merged["Acqus_Time"]),
                xlabel="13C_Glucose mMol",
                ylabel="13_C_Glucose Peak Area a.u. / sqrt(Number of scans)",
                c=merged["Time"])



plot_rel_color(merged["13C_Glucose_mMol"], merged["13C_Glucose"] / merged["Total_Area"],
                xlabel="13C_Glucose mMol",
                ylabel="13_C_Glucose Peak Area (a.u.) / Total Area (a.u.)",
                c=merged["Time"])
# test correction both by Acqus_Time and Total_Area
# (looks worse)
plot_rel_color(merged["13C_Glucose_mMol"], merged["13C_Glucose"] / merged["Total_Area"] / np.sqrt(merged["Acqus_Time"]),
                xlabel="13C_Glucose mMol",
                ylabel="13_C_Glucose Peak Area (a.u.) / Total Area (a.u.)",
                c=merged["Time"])

plot_rel_color(merged["13C_Acetate_mMol"], merged["13C_Acetate"] / merged["Total_Area"],
                xlabel="13C_Acetate mMol",
                ylabel="13_C_Acetate Peak Area (a.u.) / Total Area (a.u.)",
                c=merged["Time"])
plot_rel_color(merged["13C_Alanine_mMol"], merged["13C_Alanine"] / merged["Total_Area"],
                xlabel="13C_Alanine mMol",
                ylabel="13_C_Alanine Peak Area (a.u.) / Total Area (a.u.)",
                c=merged["Time"])
plot_rel_color(merged["13C_Alanine_mMol"], merged["13C_Alanine2"] / merged["Total_Area"],
                xlabel="13C_Alanine2 mMol",
                ylabel="13_C_Alanine2 Peak Area (a.u.) / Total Area (a.u.)",
                c=merged["Time"])
plot_rel_color(merged["13C_Butyrate_mMol"], merged["13C_Butyrate"] / merged["Total_Area"],
                xlabel="13C_Butyrate mMol",
                ylabel="13_C_Butyrate Peak Area (a.u.) / Total Area (a.u.)",
                c=merged["Time"])
plot_rel_color(merged["13C_Ethanol_mMol"], merged["13C_Ethanol"] / merged["Total_Area"],
                xlabel="13C_Ethanol mMol",
                ylabel="13_C_Ethanol Peak Area (a.u.) / Total Area (a.u.)",
                c=merged["Time"])

"""
merged_temp = merged[merged["Acqus_Time"] == 128]
plot_rel_color(merged_temp["13C_Glucose_mMol"], merged_temp["13C_Glucose"],
                xlabel="13C_Glucose mMol",
                ylabel="13_C_Glucose Peak Area a.u.",
                c=merged_temp["Time"])
"""
                
"""
# from att5_peak_match2_sliders.py
# areas are negative because ppm is decreasing
>>> real_times
array([  1., 101.,  11., 111., 121.,   2.,  21.,  31.,  41.,   5.,  51.,
        61.,  71.,  81.,  91.])
>>> areas = -np.trapz(traces, x=ppm, axis=0)
>>> areas
array([   5725.15982512,  251404.54377862, 1597985.85263905,
       1317754.22866447,  215673.8013313 ,   26750.70580491,
        363026.03883776,  382053.42846874,  636364.37455709,
          8888.56952334,  298581.14083037,  348092.51816054,
        406638.68362965,  289010.53542787, 1409229.00448348])
"""

"""
# area calculation with glucose peak exclusion
glc_bounds = [5.280111526483787, 5.340041650219163]
areas = -np.trapz(traces, x=ppm, axis=0)
# mask for ppm values *outside* the excluded range
mask = (ppm < glc_bounds[0]) | (ppm > glc_bounds[1])

# integrate only where mask is True
areas_glc = -np.trapz(traces[mask, :], x=ppm[mask], axis=0)
"""