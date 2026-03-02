import pandas as pd
import numpy as np

dfall = pd.read_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/outputs/Data7_13CGlc1_13C_and_1H/dfba_fluxes_all_13CGlc1.csv")
df13c = pd.read_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/outputs/Data7_13CGlc1_13C_only/dfba_fluxes_all_13CGlc1.csv")
df_compare = dfall.merge(df13c, on="time", suffixes=("_all", "_13c"))

# ensure both have the same columns
common_cols = dfall.columns.intersection(df13c.columns)

# compute distances for each column
distances = {
    col: np.linalg.norm(dfall[col] - df13c[col])  # Euclidean distance
    for col in common_cols
}

# convert to a ranked DataFrame
df_dist = (
    pd.DataFrame(list(distances.items()), columns=["Reaction", "Distance"])
    .sort_values("Distance", ascending=False)
    .reset_index(drop=True)
)

print(df_dist)

df_dist.to_csv("flux_comparison_10272025.csv", index=False)

threshold = 20
selected_rxns = df_dist[df_dist["Distance"] > threshold]["Reaction"]
print("Reactions above threshold:")
print(selected_rxns.tolist())

for rxn in selected_rxns:
    plt.figure(figsize=(8, 5))
    plt.plot(dfall["Time"], dfall[rxn], label="All", color="tab:blue")
    plt.plot(df13c["Time"], df13c[rxn], label="13C", color="tab:orange")
    plt.xlabel("Time")
    plt.ylabel("Flux value")
    plt.title(f"{rxn} (dist = {df_dist.loc[df_dist['Reaction']==rxn, 'Distance'].values[0]:.3f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"interesting_rxn_{rxn}.pdf")
    plt.close()












import pandas as pd
import matplotlib.pyplot as plt

# --- Helper function to load and rename time column ---
def load_flux_csv(path):
    df = pd.read_csv(path)
    # If the first column has no name, rename it to "Time"
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "Time"})
    return df

# --- Load the data ---
mean_A = load_flux_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/outputs/Data7_13CGlc1_13C_and_1H/13CGlc1_fluxes.csv")
min_A  = load_flux_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/outputs/Data7_13CGlc1_13C_and_1H/13CGlc1_fva_min.csv")
max_A  = load_flux_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/outputs/Data7_13CGlc1_13C_and_1H/13CGlc1_fva_max.csv")

mean_B = load_flux_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/outputs/Data7_13CGlc1_13C_only/13CGlc1_fluxes.csv")
min_B  = load_flux_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/outputs/Data7_13CGlc1_13C_only/13CGlc1_fva_min.csv")
max_B  = load_flux_csv("/data/local/jy1008/MA-host-microbiome/dfba_JY/outputs/Data7_13CGlc1_13C_only/13CGlc1_fva_max.csv")


# Iterate through all fluxes (skip the Time column)
for flux in mean_A.columns[1:]:
    plt.figure(figsize=(8, 5))
    
    # Comparison A
    plt.plot(mean_A["Time"], mean_A[flux], color="tab:blue", label="A mean")
    plt.fill_between(mean_A["Time"], min_A[flux], max_A[flux], color="tab:blue", alpha=0.2)
    
    # Comparison B
    plt.plot(mean_B["Time"], mean_B[flux], color="tab:orange", label="B mean")
    plt.fill_between(mean_B["Time"], min_B[flux], max_B[flux], color="tab:orange", alpha=0.2)
    
    plt.xlabel("Time")
    plt.ylabel("Flux value")
    plt.title(flux)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save as individual PDF
    plt.savefig(f"flux_{flux}.pdf")
    plt.close()







# Compare to NatChemBio Figure 3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load data
dfall = pd.read_csv(
    "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Jan302026_UGA_HRMAS_13C_Cells/dfba_results/dfba_fluxes_all_raw_1H_mixture.csv"
)

# Define multiple panels (each will become one PDF page)
panels = {
    "panel_b": {
        "rxns": ["ID_469", "ID_366", "ID_146", "ID_321"],
        "labels": ["cystathionine", "isovalerate kinase",
                   "2-methylbutyrate kinase", "isobutyrate kinase"],
    },
    "panel_c": {
        "rxns": ["ID_233", "ID_53", "ID_280"],
        "labels": ["PGK", "PFOR", "acetate kinase"],
    },
    "panel_d": {
        "rxns": ["ID_648"],
        "labels": ["hydrogenase"],
    },
    "panel_e": {
        "rxns": ["ICCoA-DHG-EB", "ID_314"],
        "labels": ["icocaprenoyl-CoA reductase", "Proline reductase"],
    },
    "panel_f": {
        "rxns": ["ID_383", "ID_251"],
        "labels": ["ethanol dehydrogenase", "butyrate kinase"],
    },
    "panel_g": {
        "rxns": ["ID_326"],
        "labels": ["acetyl-CoA synthetase"],
    },
    "panel_h": {
        "rxns": ["ATPsynth4_1", "RNF-Complex"],
        "labels": ["ATP synthase", "RNF complex"],
    },
    "panel_i": {
        "rxns": ["ALT_2abut", "ID_575"],
        "labels": ["alanine transaminase", "glutamate dehydrogenase"],
    },
}

# Create multipage PDF
with PdfPages("interesting_rxns_all_panels.pdf") as pdf:
    for panel_name, panel_data in panels.items():
        rxns = panel_data["rxns"]
        labels = panel_data["labels"]
        rxn_dict = dict(zip(rxns, labels))

        plt.figure(figsize=(8, 4))

        for rxn in rxns:
            plt.plot(dfall["Time"], dfall[rxn], label=rxn_dict[rxn])

        plt.xlabel("Time")
        plt.ylabel("Flux value")
        plt.title(panel_name)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        pdf.savefig()   # saves current figure as a new page
        plt.close()

print("Saved multi-page PDF: interesting_rxns_all_panels.pdf")
