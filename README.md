# NMR Trajectory Fitting & Dynamic FBA Constraint Pipeline

---

## 1. Purpose

This SOP documents the pipeline for converting raw, stacked HR-MAS NMR spectra into per-metabolite concentration trajectories and time-dependent flux constraints for dynamic flux balance analysis (dFBA). It covers both the single-logistic (monotonic) and double-logistic (biphasic) curve-fitting branches and the downstream dFBA runners that consume their output.

## 2. Pipeline at a Glance

The pipeline is a strict linear sequence of three stages, with a branch point after Stage 1 depending on whether a metabolite's trajectory is expected to be monotonic or biphasic.

```bash
python peak_match_global.py

# Branch 1 (single logistic):
python process_trajectories_global.py
python run_dFBA.py

# Branch 2 (double logistic):
python process_trajectories_global_multi.py
python run_dFBA_multi.py
```

## 3. Prerequisites

- Python environment with: `pandas`, `numpy`, `scipy`, `matplotlib`, `plotly`, `cobra` (COBRApy), `networkx` (+ `pygraphviz` for `graphviz_layout`), `cycler`, `pystan`/`httpstan` (imported as `stan`), `configparser`.
- Input: a preprocessed stacked-trace `.xlsx` (produced upstream, outside this SOP) and a reference-peak text file (ppm + label).
- A COBRApy JSON model file (e.g. `icdf843.json`) for the organism/community model used in Step 3. This can be obtained from https://github.com/Massachusetts-Host-Microbiome-Center/nmr-cdiff/blob/main/data/icdf843.json
- All scripts are run from the directory containing `config/`, since config paths are relative (`config/config_....ini`).

## 4. Configuration Files

Two `.ini` files drive the pipeline. Both use `configparser` with case-sensitive keys.

### 4.1 `config/config_UGA_HRMAS_13C_Cells_1H_standard.ini` - Steps 1 & 2

| Section | Key(s) | Meaning |
|---|---|---|
| `[paths]` | `working_dir`, `input_stack`, `input_ref_peaks`, `output_dir` | Location of the stacked-trace xlsx, the reference-peak list, and where Step 1 writes its outputs |
| `[params]` | `base_fit_window`, `prominence_factor`, `keep_trace_indices` | ppm half-width per peak window; peak-detection sensitivity; optional trace/timepoint masking |
| `[trajectories]` | `input_dir`, `output_dir`, `exp_name`, `plot_individual_metabs`, `overwrite_pkls` | Controls Step 2: where to read Step 1's JSON files from, where to write curve fits, and whether to force-refit cached Stan pickles |
| `[proton_num]` | one entry per metabolite | Proton count used to normalize each metabolite's raw NMR peak area |
| `[scale_mMol_to_initial]` / `[scale_mMol_to_asymptote]` / `[scale_mMol_to_dss]` | one entry per metabolite | Three mutually exclusive strategies for converting normalized areas into mMol (see 4.1.1) |

#### 4.1.1 Concentration-scaling strategies

- **`scale_mMol_to_initial`** - areas are scaled so the first timepoint equals a specified initial concentration. Use when the logistic's lower asymptote poorly represents the true starting point (e.g. the reaction has already partially progressed). (NOTE: this is currently the preferred default method, over using the asymptote.)
- **`scale_mMol_to_asymptote`** - areas are scaled so the fitted logistic's upper asymptote matches a specified concentration (initial or final).
- **`scale_mMol_to_dss`** - areas are scaled per-timepoint relative to the internal DSS reference peak, using a known DSS concentration and metabolite-specific ratio slopes.

A given metabolite should appear under exactly one of these three sections.

### 4.2 `config/config_dfba_UGA_HRMAS_13C_Cells.ini` - Step 3

| Section | Key(s) | Meaning |
|---|---|---|
| `[dfba_params]` | `input_dir`, `output_dir`, `exp_name`, `objective`, `modelfile`, `logistic_param_dir`, `steps_per_hour`, `time_range`, `aidan_bounds_dir` | `objective` is the model's objective reaction (e.g. `ATP_sink`); `modelfile` is the COBRApy JSON model; `logistic_param_dir` must point at Step 2's `trajectories/logistic_params_conc/` folder |
| `[dfba_constraints]` | one entry per constrained reaction ID | Maps a model reaction ID (e.g. `Ex_glc`, `Sec_ac`) to the logistic-parameter CSV (from Step 2) that defines its time-dependent flux bounds |
| `[dfba_bound_scale_test]` | optional, per reaction | Double-logistic branch only - multiplicative sensitivity-test scalars applied uniformly to a reaction's estimated flux |
| `[dfba_tracked_reactions]` | `ids = ` comma-separated list | Reaction IDs whose fluxes are recorded at every simulated timestep |

NOTE: Currently, `run_dFBA.py` and `run_dFBA_multi.py` both hard-code the same config filename (`config/config_dfba_UGA_HRMAS_13C_Cells.ini`), and both Step 2 scripts write into the same `trajectories/logistic_params_conc/` folder using the same file-naming convention. Confirm which branch you intend to run before executing Step 3 - running both branches back-to-back for the same metabolite will overwrite the first branch's parameter CSV with the second branch's.

## 5. Step 1 - `peak_match_global.py`

```bash
python peak_match_global.py
```

For each reference peak/metabolite listed in `input_ref_peaks`, the script:

1. Plots a colorbar overview of the traces in that peak's ppm window.
2. Opens an interactive alignment window to correct per-trace ppm drift (`edit_ppm_shifts`).
3. Opens an interactive background-peak fitter window - background peaks near the target can be fit and subtracted, or skipped.
4. Opens an interactive baseline editor on the background-subtracted data.
5. Opens an interactive signal-peak fitter window to fit the target metabolite peak and compute its area at every timepoint.

Outputs one JSON file per metabolite (`nmr_fit_global_<exp>_<label>_<ppm>.json`) with per-timepoint fitted areas plus metadata (real times, trace mask, baseline, background-fit summary), and a companion diagnostic PDF.

Already-processed metabolites (JSON already exists) are skipped automatically on re-run - delete the JSON to redo a metabolite. The optional `keep_trace_indices` config setting excludes specific timepoints/traces from fitting.

## 6. Choosing a Branch

Use the **single-logistic branch** when a metabolite's trajectory is expected to be monotonic - a single rise or fall (e.g. straightforward consumption or production). Use the **double-logistic branch** when the trajectory may be biphasic (e.g. a delayed second production or consumption phase).

The double-logistic model has a built-in shrinkage prior on the second transition's amplitude, so it degrades gracefully toward a single-logistic-like fit for metabolites that are actually monotonic - but it adds two more parameters to interpret and costs more sampling time. Also, the variance on the second logistic fit's parameters tends to be larger relative to their magnitudes than the primary logistic fit.

## 7. Step 2 - Trajectory Fitting (Bayesian Logistic Regression)

Both `process_trajectories_global.py` and `process_trajectories_global_multi.py`:

- Group Step 1's per-metabolite JSON outputs into a single per-timepoint table of raw NMR areas.
- Normalize each metabolite's raw area by its proton count (`[proton_num]`).
- Fit a Bayesian logistic curve via Stan/PyStan (NUTS sampler, 4 chains × 1000 post-warmup samples per chain) to each metabolite's trajectory in raw peak area space. Time and area are each internally rescaled to roughly $[0,1]$ before fitting and rescaled back afterward.
- Convert the fitted curves and posterior samples into concentration units (mMol) using whichever of the three `scale_mMol_to_*` strategies applies to that metabolite.
- Save raw-area posterior samples (`logistic_params/*.csv`, `*.pkl`), PDF plots of the area-space fits, concentration-space posterior samples (`logistic_params_conc/*.csv` - this is Step 3's input), and PDF plots of the concentration-space fits.

### 7.1 Single-logistic model (`process_trajectories_global.py`)

For a metabolite's trajectory $(x_i, y_i)$, $i = 1..N$, where $x$ is time and $y$ is normalized NMR area, the model is a 4-parameter logistic:

$$
y_i \sim \mathrm{Normal}\left(A + (B - A)\cdot \sigma\left(\frac{x_i - C}{D}\right),\ \sigma_{\text{noise}}\right)
$$

where $\sigma(z) = \dfrac{1}{1+e^{-z}}$ is the logistic sigmoid, and:

- $A$ - lower asymptote ($A > 0$)
- $B$ - upper asymptote ($B > 0$)
- $C$ - inflection-point time
- $D$ - signed slope/width parameter. Its sign is fixed from data as `D_sign` (the sign of the Spearman correlation between time and area, so the fit cannot flip the trajectory's direction); its magnitude $D_{\text{mag}} > 0$ carries a $Student-t(3, 0, 1)$ prior.
- $\sigma_{\text{noise}}$ - residual noise SD, with a $Student-t(3, 0, 0.1)$ prior

The sign of the logistic is pre-calculated to prevent D from crossing zero: this avoids a vanishing-gradient effect where sampled logistics tend to appear flat rather than fit the data.

Priors (on the internally rescaled axes which range [0,1]):

$$
A \sim \mathrm{Normal}(0, 0.5), \qquad B \sim \mathrm{Normal}(1, 0.5), \qquad C \sim \mathrm{Normal}(0.5, 0.5)
$$

### 7.2 Double-logistic model (`process_trajectories_global_multi.py`)

Adds a second sigmoid component to capture a possible second transition, with an additional tuneable shrinkage parameter so the model only "uses" the second phase when the data support it:

$$
y_i \sim \mathrm{Normal}\left(A + \mathrm{amp}_1\cdot \sigma\left(\frac{x_i - C_1}{D_1}\right) + \mathrm{amp}_2\cdot \sigma\left(\frac{x_i - C_2}{D_2}\right),\ \sigma_{\text{noise}}\right)
$$

- $A$ - baseline offset
- $\mathrm{amp}_1$ - first-transition amplitude; unshrunk ($\mathrm{Normal}(0, 0.5)$ prior) - the first transition is always allowed to fit
- amp2 = amp2,raw $\times \lambda_2$, where $\mathrm{amp}_{2,\text{raw}} \sim \mathrm{Normal}(0,1)$ and $\lambda_2 \sim \mathrm{HalfNormal}(0, \tau_2)$ is a local shrinkage scale (default $\tau_2 = 0.1$). This shrinks the second transition's amplitude toward zero unless the data provide evidence for it, so a purely monotonic trajectory is fit with $\mathrm{amp}_2 \approx 0$ automatically.
- $C_1, C_2$ - inflection times for the first/second transitions ($\mathrm{Normal}(0.5, 0.5)$ priors on rescaled time)
- $D_1, D_2$ - signed slope/width parameters. $D_1$'s sign is estimated from a linear fit to the first half of the trajectory; $D_2$'s sign is forced to be the opposite of $D_1$'s - i.e. the second transition is constrained to trend opposite the first (appropriate for, e.g., a fast initial rise followed by a later decline).

As with the single-logistic model, time and area are rescaled to $[0,1]$ before fitting, and posterior samples are rescaled back to original units afterward.

## 8. Step 3 - dFBA Constraint Construction & Simulation

```bash
python run_dFBA.py          # single-logistic branch
python run_dFBA_multi.py    # double-logistic branch
```

Both scripts:

- Load the COBRApy model from `modelfile`, set the objective reaction, apply a small number of hard-coded reaction-bound overrides (e.g. zeroing the `Sec_but` upper bound - review the in-script comments before reusing on a new dataset/model), and set the solver to GLPK.
- For each entry in `[dfba_constraints]`, load that metabolite's posterior-sample CSV from Step 2's `logistic_params_conc/` output, and build a time-dependent flux-bound function from the analytic derivative of the fitted logistic curve(s).
- Run a dFBA simulation (`dFBA_JY.py`) that steps through `time_range` at `steps_per_hour` resolution, applies every constraint's bounds to the model at each timestep, solves via parsimonious FBA (pFBA), and records fluxes for every reaction in `[dfba_tracked_reactions]`.

### 8.1 Flux-bound derivation

The exchange/secretion flux implied by a fitted concentration curve is its time derivative. For the single-logistic curve:

$$
\frac{dC}{dt} = \frac{B - A}{D}\cdot \sigma(z)\cdot\bigl(1-\sigma(z)\bigr), \qquad z = \frac{t - C}{D}
$$

For the double-logistic curve, the same expression is evaluated separately for $(\mathrm{amp}_1, C_1, D_1)$ and $(\mathrm{amp}_2, C_2, D_2)$ and the two components are summed:

$$
\frac{dC}{dt} = \frac{\mathrm{amp}_1}{D_1}\sigma(z_1)\bigl(1-\sigma(z_1)\bigr) + \frac{\mathrm{amp}_2}{D_2}\sigma(z_2)\bigl(1-\sigma(z_2)\bigr)
$$

At each simulated time $t$, the derivative is evaluated across all posterior samples for that metabolite; the sample mean and a chosen percent CI (95% by default) define the raw lower/upper flux bound. The raw bound is then:

- optionally widened by a `bound_scale` multiplier,
- clamped so the bound doesn't cross zero relative to the mean (`run_dFBA.py` only),
- given a small `leak` tolerance if both bounds fall within `leak_tol` of zero, to avoid over-constraining a reaction to exactly zero flux.

Uptake reactions (`Ex_*`) have their sign flipped so that consumption is represented as negative flux, matching COBRApy's exchange-reaction convention.

`run_dFBA_multi.py` additionally supports an optional `scale_factor` per reaction (via `[dfba_bound_scale_test]`) that uniformly rescales a given metabolite's estimated flux - useful for testing simulation sensitivity to concentration-estimation uncertainty (e.g. DSS reference uncertainty).

### 8.2 Outputs

| File | Contents |
|---|---|
| `dfba_fluxes_all_<exp_name>.csv` | Tracked reaction fluxes at every simulated timestep |
| `constraint_bounds.pdf` | Debug plot of each metabolite constraint's lower/upper flux bound vs. time |
| `interesting_fluxes_*.html` | Interactive Plotly plots highlighting reactions with non-trivial flux ranges |
| `dfba_results/*.pdf` | Multi-panel PDF summaries of grouped reaction fluxes |

FVA (flux variability analysis) can optionally be enabled in the `dFBA` class, but is known to stall on certain reactions in this model. A fork-based, hard-timeout FVA wrapper is used to prevent a stalled solver call from hanging the whole run; reactions known to stall (e.g. `ID_glyamintrans`, `ID_506`, `ID_357`, `ID_53`) should be added to `fva_exclude`.

## 9. Known issues

- Run the nmr preprocessing script to transform the raw data into readable spectra. This is adapted from https://github.com/Massachusetts-Host-Microbiome-Center/nmr-cdiff/tree/main, generalized to automatically process any Bruker experiment. For backward compatibility, however, this does require loading a separate conda environment (micromamba activate nmr). These should be combined into a single env.
- Peak coordinates are stored in files cfg_13C.txt and cfg_1H.txt in the data directories for backward compatibility with the preprocessing tool, but to reduce file management, these should probably be moved to the config files.
- Config file selection is hard-coded via a `config.read(...)` line near the top of each script rather than passed as a CLI argument. To point a script at a different dataset, edit that line.
- Both branches' Step 2 scripts write to the same `logistic_params/` and `logistic_params_conc/` folders using the same file-naming convention - running one branch after the other for the same metabolite overwrites the earlier branch's parameter files.
- Step 2 caches Stan fits as pickles keyed by metabolite name. `process_trajectories_global.py` always reuses an existing pickle if present; `process_trajectories_global_multi.py` honors the `overwrite_pkls` config flag (currently set to `true`, forcing a refit each run). Delete the relevant `stan_logistic_*_samples_*.pkl` file to force a specific metabolite to refit.
- matplotlib backend: scripts set `TkAgg` only on macOS (`sys.platform == "darwin"`). On a remote/cluster session, ensure a compatible backend or display is available for Step 1's interactive windows, or run Step 1 locally and Steps 2–3 on the cluster.

## 10. Output Directory Summary

| Path (relative to `output_dir`) | Produced by | Contents |
|---|---|---|
| `peak_fit_files/nmr_fit_global_*.json, *.pdf` | Step 1 | Per-metabolite fitted peak areas and diagnostic plots |
| `trajectories/logistic_params/*.csv, *.pkl` | Step 2 | Raw-area posterior parameter samples |
| `trajectories/logistic_params_conc/*.csv` | Step 2 | Concentration-scaled posterior samples (Step 3 input) |
| `trajectories/logistic_fits_raw_areas_*.pdf, logistic_fits_concs_*.pdf` | Step 2 | Diagnostic fit plots (area space and concentration space) |
| `dfba_results/dfba_fluxes_all_*.csv, constraint_bounds.pdf, interesting_fluxes_*.html` | Step 3 | Simulated flux time series and diagnostic plots |
