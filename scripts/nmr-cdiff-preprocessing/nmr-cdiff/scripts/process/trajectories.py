#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  22 11:35:45 2022

@author: Aidan Pavao for Massachusetts Host-Microbiome Center
 - Estimate concentration trajectories from HRMAS NMR time series and standard
   solutions
 - Use python 3.8+ for best results
 - See /nmr-cdiff/venv/requirements.txt for dependencies

Copyright 2022 Massachusetts Host-Microbiome Center

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from standard_curves import compute_standard_curves
from curveshapes import logi, logi_sum, logi_c, logi_flex
from get_color import get_cmap

SCDIR = os.path.dirname(__file__)   # location of script
STANDARDS_PATH = SCDIR + '/../../data/standards/'
curvedata = {
    "logistic": {
        "function": logi,
        "p0": [500, 0.2, 12],
        "scale_mask": [1, 0, 0]
    },
    "logistic_r": {
        "function": logi,
        "p0": [1000, -0.2, 12],
        "scale_mask": [1, 0, 0]
    },
    "logistic_C_r": {
        "function": logi_c,
        "p0": [1000, -0.2, 12, 400],
        "scale_mask": [1, 0, 0, 1]
    },
    "logistic_sum": {
        "function": logi_sum,
        "p0": [120, 0.6, 8, -120, 0.1, 24],
        "scale_mask": [1, 0, 0, 1, 0, 0]
    },
}

class CoeffsWriter():
    def __init__(self):
        self.popt = []
        self.perr = []
        self.sopt = []
        self.serr = []

    def add_raw(self, cpd, popt, perr):
        self.popt.append([cpd] + [f"{ele:.3f}" for ele in popt])
        self.perr.append([cpd] + [f"{ele:.3f}" for ele in perr])

    def add_scaled(self, cpd, sopt, serr):
        self.sopt.append([cpd] + [f"{ele:.3f}" for ele in sopt])
        self.serr.append([cpd] + [f"{ele:.3f}" for ele in serr])

    def _write_coeffs(self, filepath, opt, err):
        with open(filepath, "w") as wf:
            wf.write("Met,L,k,x0,C\n")
            for o, e in zip(opt, err):
                segs = [o[0]] + [f"{k}pm{m}" for (k, m) in zip(o[1:], e[1:])]
                wf.write(','.join(segs) + ','*(5-len(segs)) + '\n')

    def write_coeffs(self, filepath):
            self._write_coeffs(f"{filepath}_p.csv", self.popt, self.perr)
            self._write_coeffs(f"{filepath}_s.csv", self.sopt, self.serr)            

def fill_mask(vector, curveshape):
    if curveshape.endswith('_r'):
        cv = np.flip(vector)
        isnan = list(np.isnan(cv))
        idx = isnan.index(False)
        return [i >= len(vector) - idx for i in range(len(vector))]
    else:
        isnan = list(np.isnan(vector))
        idx = isnan.index(False)
        return [i < idx for i in range(len(vector))]
    

def fit_trajectories(filepath, substrate, init='11', plot=False, tscale=None):
    """Estimate concentration trajectories for HRMAS NMR time series.

    Parameters:
    filepath -- path to the directory containing the Bruker run. If an Excel
        spreadsheet with the output from process.py does not exist, the NMR
        experiment will be processed.
    init -- initial spectrum for processing script, if it will be run
    plot -- True to show a plot of the trajectories
    tscale -- a linear function mapping run time to scaled time. Recommended if
        multiple runs are being analyzed, can be calculated using
        synchronizers.py
    spath -- if the substrate requires a set of NMR standards to estimate
        concentration, the filepath to an experiment measuring those standards

    Returns a dictionary of scaled logistic coefficients for each metabolite,
    and dictionaries of lower and upper bounds for those coefficients from
    scipy.
    """

    coeffs = CoeffsWriter()

    basename = os.path.basename(filepath) # current run name

    print("=========")
    print(f"Computing logistic trajectories for run {basename}.")
    print("=========")

    # Load "area" sheet of excel file
    fn = f'{filepath}/{basename}_13C.xlsx'
    usecols = ["Time", substrate.name] + [met for met in substrate.products]
    areas = pd.read_excel(fn, sheet_name='area', engine='openpyxl', usecols=lambda x: x in usecols)
    detected_products = [met for met in substrate.products if met in areas.columns]
    # areas = areas.fillna(0)

    # Scale and crop time series
    time = areas['Time'].to_numpy()
    if tscale is not None:
        time = tscale(time)
    mask = (time >= 0) & (time <= 36)
    time = time[mask]
    areas = areas.loc[mask, :]

    # Perform logistic fit on products, calculate "gscale" scale factor for experiment
    signals = dict()
    curves = dict()
    curves_err = dict()

    signal = areas[substrate.name].to_numpy()
    signal[fill_mask(signal, substrate.curve)] = 0
    function = curvedata[substrate.curve]["function"]
    p0 = curvedata[substrate.curve]["p0"]
    scale_mask = curvedata[substrate.curve]["scale_mask"]
    nan_mask = ~np.isnan(signal)
    popt, pcov = scipy.optimize.curve_fit(function, time[nan_mask], signal[nan_mask], p0=p0)

    amplitude = np.sum(popt[np.array(scale_mask, dtype=bool)])
    gscale = (substrate.cx/amplitude)**np.array(scale_mask) # only scale L (and C)

    perr = np.sqrt(np.diagonal(pcov))
    coeffs.add_raw(substrate.name, popt, perr)

    # Scale signal to concentration using gscale array
    popt *= gscale
    perr *= gscale
    coeffs.add_scaled(substrate.name, popt, perr)

    curves[substrate.name] = popt
    curves_err[substrate.name] = perr
    signals[substrate.name] = signal*gscale[0]
    print("Logistic coefficients")
    print("---------")
    print_coeffs(substrate.name, popt)
    gscale = gscale[:3] # products do not have C coefficient

    # Get products curves
    for cpd in detected_products:
        scale = substrate.products[cpd]
        product_curve = curvedata[substrate.product_curves[cpd]]
        function = product_curve["function"]
        p0 = product_curve["p0"]
        scale_mask = product_curve["scale_mask"]
        signal = areas[cpd].to_numpy()
        signal[fill_mask(signal, substrate.product_curves[cpd])] = 0
        nan_mask = ~np.isnan(signal)
        try:
            popt, pcov = scipy.optimize.curve_fit(function, time[nan_mask], signal[nan_mask], p0=p0)
        except:
            print(f"Optimal parameters not found for compound {cpd} and run {basename}.")
            continue
        perr = np.sqrt(np.diagonal(pcov))
        coeffs.add_raw(cpd, popt, perr)

        # Scale curves according to Pavao et al. 2023; Methods: Estimation of exchange fluxes for dFBA
        if substrate.from_standards:
            pscale = (gscale[0]*scale)**scale_mask # From Equation 6: [Glucose]/S_glucose * a
        else:
            amplitude = np.sum(popt[np.array(scale_mask, dtype=bool)])
            pscale = (substrate.cx*scale/amplitude)**scale_mask # [Leucine]_init * Y_product / L_product
        popt *= pscale
        perr *= pscale
        coeffs.add_scaled(cpd, popt, perr)

        signals[cpd] = signal*pscale[0]
        curves[cpd] = popt
        curves_err[cpd] = perr
        print_coeffs(cpd, popt)

        ## Add curves for other oxidative BCAA fermentations
        if cpd == "Isovalerate":
            for subst, prod in [("Valine", "Isobutyrate"), ("Isoleucine", "2-methylbutyrate")]:
                iscale = (substrate.products[prod]/substrate.products[cpd])**np.array([1, 0, 0])
                popt_i = popt*iscale
                perr_i = perr*iscale
                coeffs.add_scaled(prod, popt_i, perr_i)
                curves[prod] = popt_i
                curves_err[prod] = perr_i
                vscale = (-1)**np.array([0, 1, 0])
                popt_v = popt_i*vscale
                coeffs.add_scaled(subst, popt_v, perr_i)
                curves[subst] = popt_v
                curves_err[subst] = perr_i        

    coeffs.write_coeffs(f"{filepath}/{basename}")
    plot_curves(time, signals, curves, substrate, f"{filepath}/{basename}_pan.svg")

    return curves, curves_err

def call_fit(args):
    n_args = len(args)
    if n_args > 1:
        print("Too many arguments. Please give only the run directory, or no "
              + "arguments to use the current working directory.")
    else:
        if n_args == 0:
            filepath = os.getcwd() # path to working directory
        else:
            filepath = args[0]
        return fit_trajectories(filepath, plot=True, scaled=True)

def print_coeffs(name, popt):
    print(name)
    print(f" - L : {popt[0]}")
    print(f" - k : {popt[1]}")
    print(f" - x0: {popt[2]}")
    if len(popt) > 3:
        print(f" - C : {popt[3]}")

def plot_curves(time, signals, curves, substrate, outpath):
    fig = plt.figure(figsize=(2.5, 2), constrained_layout=True)
    ax = fig.add_subplot(111)
    times = np.linspace(0, 36, num=36*10)
    cmap = get_cmap()
    plt.axhline(ls=':', color=cmap['gray'], linewidth=1, zorder=1)

    # Plot substrate
    for met, signal in signals.items():
        color = cmap[met]
        curve = curves[met]
        ax.plot(
            times, logi_flex(times, *curve), color=color, lw=1, label=met,
            zorder=2,
        )
        im = ax.scatter(
            time, signal, marker='o', color=color, facecolors=color,
            edgecolors='w', linewidths=0.5, sizes=[22 for ele in signal], zorder=3,
            label=f"_{met}",
        )

    # Format
    # afont = {'fontname': 'Times New Roman', 'size': 7}
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Estimated Concentration (mM)")
    ax.set_xlim((0, 36))
    if substrate.plot_ticks is not None:
        start = substrate.plot_ticks["start"]
        stop = substrate.plot_ticks["stop"]
        major = substrate.plot_ticks["major_step"]
        minor = substrate.plot_ticks["minor_step"]
        ax.set_yticks(np.arange(start, stop+major, major))
        ax.set_yticks(np.arange(start, stop+minor, minor), minor=True)
    ax.set_xticks([0, 12, 24, 36])
    ax.set_xticks(list(range(37)), minor=True)
    ax.set_xticklabels(ax.get_xticks())
    ax.set_yticklabels(ax.get_yticks())
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    plt.setp(ax.spines.values(), linewidth=0.5)
    # plt.legend()
    # plt.show()
    plt.savefig(outpath)
    
if __name__ == "__main__":
    call_fit(sys.argv[1:])
