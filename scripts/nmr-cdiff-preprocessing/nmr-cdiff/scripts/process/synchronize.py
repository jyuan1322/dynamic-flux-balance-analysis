#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  03 10:56:20 2022

@author: Aidan Pavao for Massachusetts Host-Microbiome Center
 - Synchronize multiple NMR runs using 1H spectra
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
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from curveshapes import logi_c, inv_logi_c

def calculate_5p(filepath, plot=False):
    """Calculate x value when logistic reaches min/max 5 percent."""
    basename = os.path.basename(filepath) # current run name

    # Process 1H stack if necessary, will be written to an excel file
    if not os.path.exists(f'{filepath}/{basename}_1H.xlsx'):
        if basename == '20210519_13CGlc':
            init = '51'
        else:
            init = '11'
        # process('1H', init, False)

    # Load "area" sheet of excel file
    areas = pd.read_excel(
        f'{filepath}/{basename}_1H.xlsx', sheet_name='area',
        engine='openpyxl', usecols=lambda x: x not in ['Scans'],
    )
    # Identify reference peak and calculate curve fit
    if "13CLeu" in basename:
        # Peak is split in 13C-Leu condition, so the shift must be adjusted
        refpk = "Isocaproate 0.76"
        tsh = 18
    else:
        refpk = "Isocaproate 0.8642"
        tsh = 10
    mask = areas['Time'] <= 36
    T = np.array(areas['Time'])[mask]
    S = np.array(areas[refpk])[mask]
    popt, pcov = scipy.optimize.curve_fit(
        logi_c, T, S, p0=[max(S), 0.3, tsh, 0]
    )
    L = popt[0]
    C = popt[3]
    popt[0] = 1
    popt[3] = 0
    c = basename
    P = inv_logi_c(np.array([0.05, 0.95]), *popt)
    if plot:
        ax = plt.axes()
        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(xx, logi_c(xx, *popt), color=color, lw=1, label=basename)
        ax.scatter(T, (S - C)/L, marker='o', c=color,
                   facecolors=color, edgecolors='w', linewidths=0.5,
                   label=f"_{basename}")
        plt.show()
    return P

def get_shift(P, R, stretch=True):
    """Return function to shift timescale"""
    if stretch:
        return lambda x: (R[1] - R[0])/(P[1] - P[0])*(x - P[0]) + R[0]
    else:
        return lambda x: x - P[0] + R[0]

def synchronizers(runs, stretch=True, plot=False):
    """Synchronize spectra for downstream analyses.

    Parameters:
    runs -- iterable containing filepaths of each run to be synchronized; the
        first run will be the reference to which the other runs are shifted.
    stretch -- boolean flag to normalize the time scales by metabolic rate

    Generates linear functions that transform the time vector for each run,
        using the first run as the basis.
    """
    if plot:
        xx = np.linspace(0, 36, num=36*10)

    for ii, filepath in enumerate(runs):
        P = calculate_5p(filepath, plot=plot)
        basename = os.path.basename(filepath) # current run name
        if ii == 0:
            R = P
            yield lambda x: x
        else:
            print(f"{basename}: scaling window {P} to {R}.")
            yield get_shift(P, R, stretch=stretch)

def adjust_curve(R_file, P_coeffs, coeffs, stretch=True):
    """Synchronize a curve that has already been calculated."""
    R = calculate_5p(R_file, plot=False)
    P = inv_logi_c(np.array([0.05, 0.95]), *P_coeffs)
    if stretch:
        A = (R[1]-R[0])/(P[1]-P[0])
    else:
        A = 1
    xshift = get_shift(P, R, stretch=stretch)
    return np.array([coeffs[0], A*coeffs[1], xshift(coeffs[2]), coeffs[3]])

if __name__ == "__main__":
    synchronizers([fpath for fpath in sys.argv[1:]])
