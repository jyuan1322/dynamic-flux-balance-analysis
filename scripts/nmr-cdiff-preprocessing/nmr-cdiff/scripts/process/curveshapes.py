#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  22 11:57:02 2022

@author: Aidan Pavao for Massachusetts Host-Microbiome Center
 - Logistic function and variations
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

import collections

import numpy as np
from matplotlib import pyplot as plt
class Metabolite:
    """Class to hold LogisticSet objects for compounds. Allows multiple substrates."""
    def __init__(self, model_id, name):
        self.name = name
        self.id = model_id
        self.scale_map = {}
        self.logistic_sets = collections.defaultdict(LogisticSet)

    @classmethod
    def sum_solutions(cls, solutions):
        val = np.sum(np.array([sol[0] for sol in solutions]), axis=0)
        err = rss(np.array([sol[1] for sol in solutions]), axis=0)
        dval = np.sum(np.array([sol[2] for sol in solutions]), axis=0)
        derr = rss(np.array([sol[3] for sol in solutions]), axis=0)
        return val, err, dval, derr

    def set_substrate_scale(self, substrate, scale):
        self.scale_map[substrate] = scale

    def get_substrate_scale(self, substrate):
        return self.scale_map.get(substrate, 1.)

    def add_curve(self, substrate, params, errors):
        self.logistic_sets[substrate].add_curve(params, errors)

    def eval(self, ts, substrate=None):
        if substrate is not None:
            return self.logistic_sets[substrate].eval(ts)
        solutions = [ls.eval(ts) for ls in self.logistic_sets.values()]
        return self.sum_solutions(solutions)

    def get_sol(self, t, substrate=None):
        if substrate is not None:
            return self.logistic_sets[substrate].get_sol(t)
        solutions = [ls.get_sol(t) for ls in self.logistic_sets.values()]
        return self.sum_solutions(solutions)
    
    def get_bounds(self, t, substrate=None):
        if substrate is not None:
            return self.logistic_sets[substrate].get_bounds(t)
        val, err, dval, derr = self.get_sol(t, substrate=None)
        exch_l = dval - 1.96*derr
        exch_u = dval + 1.96*derr
        signal_l = val - 1.96*err
        signal_u = val + 1.96*err
        return exch_l, exch_u, signal_l, signal_u
    
    def halfmax(self, substrate):
        """Return timepoint of maximum flux (logistic half-max)."""
        return self.logistic_sets[substrate].halfmax()
    
    def avg_x0(self, substrate):
        return self.logistic_sets[substrate].avg_x0()
    
    def tshift(self, substrate, t):
        self.logistic_sets[substrate].tshift(t)

    def set_runcount(self, substrate, runcount):
        self.logistic_sets[substrate].set_runcount(runcount)

    def display(self):
        for substrate, ls in self.logistic_sets.items():
            print(f"Substrate: {substrate}")
            print("="*len(substrate))
            ls.display()
            print()
    
    def display_avg_coeffs(self):
        for substrate, ls in self.logistic_sets.items():
            print(f"Substrate: {substrate}")
            print("="*len(substrate))
            ls.display_avg_coeffs(prefix="\t")
            print()

class LogisticSet:
    """Class to store coefficients for multiple logistic equations.
    Equations will be evaluated independently and solutions averaged (with
    error propagation).
    """
    param_names = ["L", "k", "x0", "L2 (C)", "k2", "x02"]
    def __init__(self):
        """Init set -- curves and errors are lists of coefficients/errs."""
        self.curves = []
        self.errors = []
        self.scalar = 1. # scale factor for logistic
        self.time = None # time scale to evaluate
        self.val = None # evaluated function over time series
        self.err = None # evaluated errors over time series
        self.dval = None # evaluated derivative over time series
        self.derr = None # evaluated derivative error over time series

    def add_curve(self, params, errors):
        self.curves.append(params.tolist())
        self.errors.append(errors.tolist())

    def _eval(self, x, eval_fn, err_fn, avg_first=True):
        """Internal evaluate with eval_fn and error function err_fn."""
        if avg_first:
            val, err = self._eval_avg_first(x, eval_fn, err_fn)
        else:
            val, err = self._eval_avg_last(x, eval_fn, err_fn)
        return self.scalar*val, err

    def _eval_avg_last(self, x, eval_fn, err_fn):
        """Internal evaluate with eval_fn and error function err_fn.
        Evaluates each curve in self._curves independently, and averages the
        result.
        """
        eval_sol = np.array([eval_fn(x, *ps) for ps in self.curves])
        eval_err = np.array([err_fn(x, *ps, *es) for ps, es in zip(self.curves, self.errors)])
        avg_sol = np.sum(eval_sol, axis=0)/eval_sol.shape[0]
        avg_err = np.sqrt(np.sum(np.square(eval_err), axis=0))/eval_err.shape[0]
        # avg_err = rss(eval_sol, axis=0)/eval_err.shape[0]
        return avg_sol, avg_err

    def _eval_avg_first(self, x, eval_fn, err_fn):
        """Internal evaluate with eval_fn and error function err_fn.
        Averages each parameter accross all curves, then evaluates using the
        average coefficient values.
        """
        avg_ps = np.average(np.array(self.curves), axis=0, keepdims=True)[0]
        avg_es = rss(np.array(self.errors), axis=0)
        avg_sol = eval_fn(x, *avg_ps)
        avg_err = err_fn(x, *avg_ps, *avg_es)
        return avg_sol, avg_err

    def eval(self, ts):
        """Evaluate and store function and derivative over time series ts."""
        self.time = ts
        self.val, self.err = self._eval(ts, logi_flex, err_logi_flex)
        self.dval, self.derr = self._eval(ts, ddx_logi_flex, err_ddx_logi_flex)
        return self.val, self.err, self.dval, self.derr

    def get_sol(self, t):
        """Get solution at time t if stored, else calculate it."""
        idx = np.nonzero(self.time == t)
        if idx[0].size == 0:
            val, err = self._eval(t, logi_flex, err_logi_flex)
            dval, derr = self._eval(t, ddx_logi_flex, err_ddx_logi_flex)
            return val, err, dval, derr
        else:
            i = idx[0][0]
            return self.val[i], self.err[i], self.dval[i], self.derr[i]

    def get_bounds(self, t):
        """Get 95% confint bounds at time t if stored, else calculate them."""
        val, err, dval, derr = self.get_sol(t)
        exch_l = dval - 1.96*derr
        exch_u = dval + 1.96*derr
        signal_l = val - 1.96*err
        signal_u = val + 1.96*err
        return exch_l, exch_u, signal_l, signal_u

    def halfmax(self):
        """Return timepoint of maximum flux (logistic half-max)."""
        if sum([curve[1] for curve in self.curves]) < 0:
            return self.time[np.argmin(self.dval)]
        else:
            return self.time[np.argmax(self.dval)]
        
    def avg_x0(self):
        return np.average([ps[2] for ps in self.curves])

    def tshift(self, t):
        """Shift all curves' half-max (x0) to t."""
        for param_set in self.curves:
            param_set[2] = t
        self.eval(self.time)

    def set_runcount(self, runcount):
        """Calculate scalar from total run count."""
        self.scalar = len(self.curves)/runcount

    def display(self):
        print(self.curves)
        print(self.errors)

    def display_avg_coeffs(self, prefix="\t"):
        """Display average coefficients p/m error. Pass in coefficient names."""
        avg_ps = np.average(np.array(self.curves), axis=0)
        avg_es = rss(np.array(self.errors), axis=0)
        for i, (par, err) in enumerate(zip(avg_ps, avg_es)):
            print(f"{prefix}{self.param_names[i]}: {par:.3f} ± {err:.3f}")

def rss(args, axis=None):
    """Calculate the square root of the sum of squares."""
    # return np.sqrt(np.sum(np.square(np.array(args)), axis=axis))
    return np.sqrt(np.sum(np.array(args)**2, axis=axis))

def logi(x, L, k, x0):
    """Evaluate the logistic function at point x."""
    return L*np.reciprocal(1. + np.exp(-1*k*(x - x0)))

def logi_c(x, L, k, x0, C):
    """Evaluate the logistic function at point x with vertical shift."""
    return logi(x, L, k, x0) + C

def logi_sum(x, L1, k1, x01, L2, k2, x02):
    """Evaluate the sum of two logistic functions at point x."""
    return logi(x, L1, k1, x01) + logi(x, L2, k2, x02)

def logi_flex(x, *argv):
    """Evaluate the (sum of) logistic function at point x."""
    L = argv[0]
    k = argv[1]
    x0 = argv[2]
    sol = logi(x, L, k, x0)
    if len(argv) == 3:
        return sol # typical solution
    elif len(argv) == 4:
        return sol + argv[3] # + C
    elif len(argv) == 5:
        return sol - logi_flex(x, L, *argv[3:]) # L2 = -L1
    else:
        return sol + logi_flex(x, *argv[3:])  # multiple curves

def ddx_logi(x, L, k, x0):
    """Evaluate the derivative of the logistic function at point x."""
    return k*L*np.exp(-k*(x - x0))/(1. + np.exp(-1*k*(x - x0)))**2

def ddx_logi_flex(x, *argv):
    """Evaluate the derivative of the (sum of) logistic function at point x."""
    L = argv[0]
    k = argv[1]
    x0 = argv[2]
    sol = ddx_logi(x, L, k, x0)
    if len(argv) <= 4:
        return sol  # typical solution
    elif len(argv) == 5:
        return sol - ddx_logi_flex(x, L, *argv[3:]) # L2 = -L1
    else:
        return sol + ddx_logi_flex(x, *argv[3:])  # multiple curves

def inv_logi_c(x, L, k, x0, C):
    """Evaluate the inverse of the logistic function at point x."""
    return -1/k*np.log(L/(x - C) - 1) + x0

def dydL(x, L, k, x0, C=0):
    """Logistic function partial derivative with respect to L."""
    return 1/(1 + np.exp(-k*(x - x0)))

def dydk(x, L, k, x0, C=0):
    """Logistic function partial derivative with respect to k."""
    return L*(x - x0)*np.exp(-k*(x - x0))/(1. + np.exp(-k*(x - x0)))**2

def dydx0(x, L, k, x0, C=0):
    """Logistic function partial derivative with respect to x0."""
    return -L*k*np.exp(-k*(x - x0))/(1. + np.exp(-k*(x - x0)))**2

def dydC(x, L, k, x0, C=0):
    """Logistic function partial derivative with respect to C."""
    return np.ones_like(x)

def err_logi(x, L, k, x0, C, del_L, del_k, del_x0, del_C):
    """Evaluate standard error of logistic function at point x.

    Parameters:
    L, k, x0, C -- logistic function coefficients
    del_L, del_k, del_x0, del_C -- standard error in logistic coefficients
    """
    coeffs = [L, k, x0, C]
    L_term = (dydL(x, *coeffs)*del_L)**2
    k_term = (dydk(x, *coeffs)*del_k)**2
    x0_term = (dydx0(x, *coeffs)*del_x0)**2
    C_term = (dydC(x, *coeffs)*del_C)**2
    return np.sqrt(L_term + k_term + x0_term + C_term)

def err_logi_flex(x, *argv):
    """Evaluate standard error of logistic function at point x.
    Handles case where the coefficient C is not provided (assume 0).
    """
    if len(argv) == 6:
        return err_logi(x, *argv[:3], 0, *argv[3:], 0)
    elif len(argv) == 8:
        return err_logi(x, *argv)
    elif len(argv) == 12:
        return rss((err_logi(x, *argv[0:3], 0, *argv[6:9], 0), err_logi(x, *argv[3:6], 0, *argv[9:12], 0)), axis=0)
    else:
        return None

def dzdL(x, L, k, x0):
    """Partial derivative of logistic functn. derivative with respect to L."""
    return ddx_logi(x, L, k, x0)/L

def dzdk(x, L, k, x0):
    """Partial derivative of logistic functn. derivative with respect to k."""
    da = 2*(x - x0)*np.exp(-k*(x - x0))/(1 + np.exp(-k*(x - x0)))
    db = x0 - x
    dc = 1/k
    return ddx_logi(x, L, k, x0)*(da + db + dc)

def dzdx0(x, L, k, x0):
    """Partial derivative of logistic functn. derivative with respect to x0."""
    da = 2*k*np.exp(-k*(x - x0))/(1 + np.exp(-k*(x - x0)))
    db = -k
    return ddx_logi(x, L, k, x0)*(da + db)

def compute_ddx_err_terms(x, L, k, x0, del_L, del_k, del_x0):
    coeffs = [L, k, x0]
    L_term = (dzdL(x, *coeffs)*del_L)**2
    k_term = (dzdk(x, *coeffs)*del_k)**2
    x0_term = (dzdx0(x, *coeffs)*del_x0)**2
    return L_term, k_term, x0_term

def err_ddx_logi(x, L, k, x0, C, del_L, del_k, del_x0, del_C):
    """Evaluate standard error of logistic function derivative at point x.
    Ignores coefficient C.

    Parameters:
    L, k, x0, C -- logistic function coefficients
    del_L, del_k, del_x0, del_C -- standard error in logistic coefficients
    """
    err_terms = compute_ddx_err_terms(x, L, k, x0, del_L, del_k, del_x0)
    return np.sqrt(np.sum(err_terms, axis=0))

def err_ddx_logi_flex(x, *argv):
    """Evaluate standard error of logistic function derivative at point x.
    Handles case where the coefficient C is not provided (assume 0).
    """
    if len(argv) == 6:
        return err_ddx_logi(x, *argv[:3], 0, *argv[3:], 0)
    elif len(argv) == 8:
        return err_ddx_logi(x, *argv)
    elif len(argv) == 12:
        err_terms = [
            *compute_ddx_err_terms(x, *argv[0:3], *argv[6:9]),
            *compute_ddx_err_terms(x, *argv[3:6], *argv[9:12])
        ]
        return np.sqrt(np.sum(err_terms, axis=0))
    else:
        return None
