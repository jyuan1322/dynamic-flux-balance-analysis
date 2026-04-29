import os
import pandas as pd

input_dir = "/data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Jan302026_UGA_HRMAS_13C_Cells/trajectories/logistic_params_conc"

# Read the CSV
df = pd.read_csv(os.path.join(input_dir, "logistic_params_samples_raw_1H_standard_NT_Isovalerate.csv"))

# Swap columns A and B
df[["A", "B"]] = df[["B", "A"]]

# Write to a new CSV
# df.to_csv(os.path.join(input_dir, "logistic_params_samples_raw_1H_standard_pseudo_NT_Isobutyrate.csv"), index=False)
# df.to_csv(os.path.join(input_dir, "logistic_params_samples_raw_1H_standard_pseudo_NT_2-aminobutyrate.csv"), index=False)




# Read the CSV
df = pd.read_csv(os.path.join(input_dir, "logistic_params_samples_raw_1H_standard_13C_Glucose.csv"))

# Swap columns A and B
df[["A", "B"]] = df[["B", "A"]]

# Write to a new CSV
df.to_csv(os.path.join(input_dir, "logistic_params_samples_raw_1H_standard_pseudo_ID_325.csv"), index=False)

# Read the CSV
df = pd.read_csv(os.path.join(input_dir, "logistic_params_samples_raw_1H_standard_NT_Threonine.csv"))

# Swap columns A and B
df[["A", "B"]] = df[["B", "A"]]

# Write to a new CSV
df.to_csv(os.path.join(input_dir, "logistic_params_samples_raw_1H_standard_pseudo_2HBD.csv"), index=False)