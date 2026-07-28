import configparser
from process import Stack

# read from config file
config = configparser.ConfigParser()
config.optionxform = str   # <-- turn off lowercasing
# config.read("config/config_preprocessing_UGA_HRMAS_1H.ini")
config.read("config/config_preprocessing_UGA_HRMAS_13C.ini")

input_dir = config["paths"]["input_dir"]
output_dir = config["paths"]["output_dir"]
experiment_type = config["experiment"]["spectra_type"]
first_fid = config["experiment"]["first_fid"]

s = Stack(input_dir, experiment_type, first_fid)
s.calibrate(overwrite=False)
s.process_fids(overwrite=True)
s.ridgetrace_fids(plot=False)
s.write_stack(outdir=output_dir, from_ridges=True)
