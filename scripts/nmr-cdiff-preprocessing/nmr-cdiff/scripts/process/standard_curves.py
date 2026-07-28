#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  21 16:57:24 2022

@author: Aidan Pavao for Massachusetts Host-Microbiome Center
 - Compute compound-specific 13C-NMR signal enhancement ratios
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

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

carbons = {
    'Glucose': 6,
    'Butyrate': 3,
    'Alanine': 2,
    'Ethanol': 2,
    'Acetate': 1,
    'Threonine': 3,
    '2-aminobutyrate': 3,
    '2-hydroxybutyrate': 3,
    'Propionate': 2,
    'n-Propanol': 3,
}

def compute_standard_curves(filepath=None, substrate="Glucose", plot=True, plot_confint=False):
    """Calcualte standard curve regressions from standard solutions.
    Returns dict of scale factors transforming substrate concentration to
    product concentration.

    Parameters:
    - filepath: Path to directory containing processed standard curves. If
        None, will use the working directory (default: None).
    - plot: Boolean flag to plot standard curves (default: True).
    - plot_confint: Boolean flag to include 95% confidence band of slope on
        the plot (default: False).

    Returns a scale factor for each metabolite, representing the reciprocal of
    the relative 13C signal enhancement of the metabolite with respect to the
    substrate.
    """
    if filepath is None:
        filepath = os.getcwd() # path to working directory
    basename = os.path.basename(filepath) # current run name

    # Load "area" sheet of excel file
    areas = pd.read_excel(
        f'{filepath}/{basename}_13C.xlsx', sheet_name='area',
        engine='openpyxl', usecols=lambda x: x not in ['Time', 'Scans'],
    )

    # Load file containing concentrations of each sample, same shape as areas
    conc = pd.read_excel(
        f'{filepath}/Concentrations.xlsx', engine='openpyxl',
        usecols=lambda x: x not in ['Sample'],
    )

    # Normalize concentration/area
    norm = conc/areas
    norm_products = norm[[c for c in norm.columns if c != substrate]]
    M = np.array(norm[substrate])[:, np.newaxis]

    # Linear regression of each compound's area with respect to glucose
    scale_factors = dict() # Linear regression slopes to return

    if plot:
        # plt.rc('font', **{'family':'sans-serif','sans-serif':['Times New Roman'],
        #        'size':12})
        shape = int(np.ceil(norm_products.shape[1]/2))
        fig, axs = plt.subplots(nrows=2, ncols=shape, figsize=(7.5*shape,11))

    for i, c in enumerate(norm_products.columns):
        # Filter out nan values and low signal
        if '13CThr' in filepath:
            threshold = 35
        else:
            threshold = 15
        mask = (~np.isnan(norm_products[c])) & (areas[c]/carbons[c] > threshold)
        A = np.array(norm_products[c])[mask]
        M_f = M[mask, :]

        # Compute linear regression
        slope, ssr = np.linalg.lstsq(M_f, A, rcond=None)[:2]
        slope = slope[0]
        ssr = ssr[0]
        scale_factors[c] = slope
        if c == "2-hydroxybutyrate":
            scale_factors["2-aminobutyrate"] = slope

        # Calculate and report statistics
        ser = np.sqrt(ssr/(np.sum(mask) - 2))
        sst = A.size * A.var() #np.sum((A - A.mean())**2)
        rsquared = 1 - ssr/sst
        print(c)
        print(f" - Slope: {slope}")
        print(f" - Sum of Squared Residuals: {ssr}")
        print(f" - Std. Err. Regression: {ser}")
        print(f" - R-squared: {rsquared}")

        # Plot result
        if plot:
            ax = axs[i // shape, i % shape]
            xmax = 1.1*max(norm.loc[~np.isnan(norm_products[c]), substrate])
            if plot_confint:
                pars = {'color':'k', 'ls':':', 'lw':0.5, 'alpha':0.1}
                ax.fill_between(
                    (0, xmax), (0, (slope+1.96*ser)*xmax),
                    y2=(0, (slope-1.96*ser)*xmax), **pars
                )
            ax.plot((0, xmax), (0, slope*xmax), lw=1, color='k')
            im = ax.scatter(
                np.array(norm[substrate])[mask], A, marker='o', edgecolors='w',
                cmap='rainbow_r', c=(areas[c]/carbons[c])[mask],
                facecolors=areas[c][mask], linewidths=0.5
            )
            ax.scatter(
                np.array(norm[substrate])[~mask],
                np.array(norm_products[c])[~mask], marker='x', facecolors='k',
                linewidths=1
            )
            plt.colorbar(ax=ax, mappable=im, label=f'{c} Signal Intensity')
            ax.xaxis.set_tick_params(width=1)
            ax.yaxis.set_tick_params(width=1)
            plt.setp(ax.spines.values(), linewidth=1)
            ax.set_title(c, size=14, weight='bold')
            ax.set_xlabel(
                f'[{substrate}]/Integrated {substrate} signal (mM/a.u.)',
            )
            ax.set_ylabel(f'[{c}]/Integrated {c} signal (mM/a.u.)')
            labs = f"$y = {slope:.2f}*x$\n$R^2 = {rsquared:.3f}$"
            ax.text(xmax*0.75, slope*xmax*0.75, labs, ha='left', va='top')

    if plot:
        plt.tight_layout()
        plt.savefig(f"{filepath}/{basename}_standards.svg")
        plt.close()

    # Return LinReg slopes, these scale glucose-normalized signal to Cx
    return scale_factors

if __name__ == "__main__":
    compute_standard_curves(plot=True, plot_confint=False)
