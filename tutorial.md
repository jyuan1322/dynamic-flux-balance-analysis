# NMR Analysis pipeline

This tutorial walks through processing of an NMR time series experiment intended for studying the dynamics of C. diff metabolism. The steps are:

* Process raw NMR runs into a xlsx file containing spectra
* Deconvolve selected NMR peaks and calculate areas using an interactive tool
* Perform logistic curve fitting on raw areas over time and convert these trajectories into concentrations
* Run Dynamic Flux Balance Analysis (dFBA) using constraints derived from confidence intervals on logistic trajectories.

---

## Prerequisites

### Software
```bash
micromamba create -f pystan-env.yml
micromamba create -f nmr-env.yml
```

---

## Project Structure

A typical project directory looks like this:

```text
nmr-time-series/
├── scripts
│   └── config.ini
│   └── peak_match.py
│   └── process_trajectories.py
│   └── run_dFBA.py
├── data/
│   └── Data7_13CGlc1
│       └── cfg_1H.txt
│       └── cfg_13C.txt
│       └── Data7_13CGlc1_1H.xlsx
│       └── Data7_13CGlc1_13C.xlsx
├── output/
│   └── Data7_13CGlc1
│       └── peak_fit_files
│           └── nmr_fit_Data7_13CGlc1_1H_13C_Glucose_5.3813_0.json
│           └── nmr_fit_Data7_13CGlc1_1H_13C_Glucose_5.3813_0.pdf
│           └── nmr_fit_Data7_13CGlc1_1H_13C_Glucose_5.3813_1.json
│           └── nmr_fit_Data7_13CGlc1_1H_13C_Glucose_5.3813_1.pdf
│           └── ...
│       └── logistic_params
│           └── logistic_params_samples_Data7_13CGlc1_1H_13C_Glucose.csv
│           └── stan_logistic_samples_test_glc1H_13C_Glucose.pkl
│       └── logistic_params_conc
│           └── logistic_params_samples_Data7_13CGlc1_1H_13C_Glucose.csv
│       └── dfba_results
│           └── dfba_13CGlc1_fluxes.csv
│           └── dfba_13CGlc1_fva_min.csv
│           └── dfba_13CGlc1_fva_max.csv
│           └── interesting_fluxes.html
```

---

## Procedure

### Preprocessing
Run the [nmr-cdiff](https://github.com/Massachusetts-Host-Microbiome-Center/nmr-cdiff) procedure for processing the NMR spectra into a single xlsx file. Prior to this, activate the environment with

```bash
micromamba activate nmr-env
```

### Peak Fitting

Run the following to automatically read parameters from the config file.
```bash
micromamba deactivate nmr-env
micromamba activate pystan-env
python peak_match.py
```

The relevant parameters in the config file are the below:

```bash
[paths]
working_dir = /data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/data/Data7_13CGlc1_Test
input_stack = Data7_13CGlc1_1H.xlsx
input_ref_peaks = cfg_1H.txt

[params]
base_fit_window=0.04
prominence_factor=0.1
```

`input_stack` specifies the xlsx output from the spectra preprocessing step. `input_ref_peaks` is a file storing metabolite names and peak locations in ppm in the format

```bash
5.3131  13C_Glucose
3.5528  Glycine
3.2338  Arginine
```

The parameter `base_fit_window` controls the initial viewing boundary of the peak. This may need to be adjusted for particularly wide or narrow peaks. For 1H, 0.04 typically works well, whereas 0.4 tends to work well for 13C. `prominence_factor` controls the sensitivity for detecting the number of peaks in the deconvolution - this not frequently changed.

For each metabolite in `input_ref_peaks`, a plot of all fids in the experiment colored by time point is shown, as well as a window for adjusting the left and right boundaries of the peak on the ppm axis. The deconvolution and area calculation of the peak will be applied to the bounded peak. Adjust the sliders to set the boundaries: these will be kept the same for subsequent fids, but can be adjusted if the peak shifts on the ppm axis over time.

For each fid, press the button the perform the deconvolution and area estimation. If the fit is suboptimal, press the button again. To keep a fit, close the window, and a window for fitting the next fid will appear. Upon closing the window, a json file storing the fit parameters and a pdf visualizing the fit are generated. The random seed generating the selected fit is also stored. If this script is interrupted, it will continue from the next fid for which there is no json file. If you wish to rerun an fid, simply delete the json and pdf file.

### Trajectory fitting

Run `process_trajectories.py`, which draws from the same config file. The input file and experiment name are stored in the following parameters. The experiment name should match that recorded in the json files of the prior area fit step.

```bash
[trajectories]
input_dir = /data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data7_13CGlc1_1H_Test
exp_name = Data7_13CGlc1_1H
```

The raw areas must be scaled by the proton numbers associated with the peaks.

```bash
[proton_num]
13C_Glucose = 0.5
13C_Acetate = 2.0
13C_Butyrate = 3.0
```

To convert nmr areas to concentrations, three strategies are offered:
* `scale_mMol_to_initial`: Simply set the first recorded area to a specified initial concentration
* `scale_mMol_to_asymptote`: Set the upper asymptote of the logistic fit to a specified initial concentration
* `scale_mMol_to_ratio`: Applies to glucose products whose areas are calculated from standard curves. The specified value is the slope of the metabolite's ratio of concentration over NMR area to that of glucose.

```bash
[scale_mMol_to_initial]
13C_Glucose = 27.5

[scale_mMol_to_asymptote]
Arginine = 1.15
Histidine = 0.65
Methionine = 1.34
...

# [y]/(area(y)) = m [glc]/(area(glc))
[scale_mMol_to_ratio]
13C_Acetate = 1.47302042086492
13C_Alanine = 0.2492142688906
...
```

The logistic curve fitting is conducted in stan, and to generate confidence intervals, 1000 curves are fit. The output of this step is a set of csv files, both in raw area and concentrations, which store the parameters of the 1000 fitted curves. The pkl files store the full stan objects. 

### dFBA

Run `run_dFBA.py`, which draws from the same config file. The primary input is the set of metabolites to constrain the model: each csv file is generated by the previous trajectory fitting step and stores the fitted logistic parameters scaled to concentrations in mMol. Exchange reactions beginning with "Ex_" have the sign of their fluxes flipped, whereas secretion reactions beginning with "Sec_" do not. 

```bash
[dfba_params]
logistic_param_dir = /data/local/jy1008/MA-host-microbiome/dfba_JY/nmr_area_estimation/output/Data7_13CGlc1_1H_Test/logistic_params_conc
...

[dfba_constraints]
Ex_glc = logistic_params_samples_Data7_13CGlc1_1H_13C_Glucose.csv
Sec_ac = logistic_params_samples_Data7_13CGlc1_1H_13C_Acetate.csv
...
```

The output of the dFBA step comprises csv files storing the mean flux and the min and max FVA bounds. The reactions listed under `dfba_tracked_reactions` are listed in these files. Multiple browsable html files are generated combining reactions at different levels of scale on the flux axis for easier visualization. Fluxes for certain reactions are excluded here if they do not pass thresholds for magnitude and change over time. 

---

## References

* [A Pavao et al. *Nat Chem Biol* (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10154198)
* [A Ebrahim et al. *BMC Syst Biol* (2013)](https://cobrapy.readthedocs.io/en/latest/faq.html)
* [A Riddell et al. (2021)](https://pypi.org/project/pystan/)

---
