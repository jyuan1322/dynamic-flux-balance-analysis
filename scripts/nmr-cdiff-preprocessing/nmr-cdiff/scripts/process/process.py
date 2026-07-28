"""Created on Thu Apr  8 12:00:24 2021

@author: Aidan Pavao for Massachusetts Host-Microbiome Center
 - Process raw Bruker NMR files into Excel spreadsheet for downstream analysis
 - Use python 3.8+ for best results
 - See /nmr-cdiff/venv/requirements.txt for dependencies

Copyright 2021-2022 Massachusetts Host-Microbiome Center

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

import argparse
import collections
import datetime
import glob
import json
import os
import subprocess
import traceback

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt
import nmrglue as ng
import numpy as np
import pandas as pd
import scipy

from man_bl import Baseline
from lineshapes_addon import split_lorentz_fwhm, gumbel_hb
from ui import Peak, PeakFitWindow, ConflictWindow, CalibrateWindow

from itertools import cycle

SCDIR = os.path.dirname(__file__)   # location of script
CURVE_FIT_RESOLUTION = 2**19

isotope_params = ["id", "name", "fit_ppm_delta", "fit_fwhm_max", "fit_fwhm_min",
                  "peak_pick_msep", "ppm_min", "ppm_max", "assignment_display_window",
                  "assignment_cluster_msep", "fit_cluster_msep", "fit_reach"]
Isotope = collections.namedtuple("Isotope", isotope_params)
isotopes = {}
with open(f"{SCDIR}/isotopes.json", "r") as rf:
    isotope_data = json.loads(rf.read())
    for iso, params in isotope_data.items():
        isotopes[iso] = Isotope(id=iso, **params)

def match_pulprog(isotope, pulprog, fid=None):
    # original pulprog
    if isotope == '1H':
        return pulprog == 'noesypr1d'
    elif isotope == '13C':
        # return pulprog == 'zgpg30'
        return pulprog == 'zgpg'
    elif isotope == '1H_13C':
        return pulprog == 'hsqcphpr'
    # 1H experiment 21, 26, 31, 36
    elif isotope == "1H_standard":
        return pulprog == "zgpr"
    # 1H experiment 25
    elif isotope == "1H_mixture":
        # 25, 30, 35, 40
        if fid is not None and fid % 5 == 0:
            return pulprog == "hsqcphpr"
    return False

def prompt_continue():
    while True:
        cont = input("Continue? (y/n): ")
        if cont.strip() in ['y', 'n']:
            break
        else:
            print("\tInvalid input.")
    if cont == 'n':
        return False
    else:
        return True

def rmse(vector):
    """Calculate the Root Mean Squared Error (RMSE) of a 1D-array."""
    return np.sqrt(((vector - vector.mean()) ** 2).mean())

def shift_params(s, *args):
    """Shift parameters by a factor s.
    Only acts on first element, which is usually the ppm shift.
    """
    for params in args:
        for param in params:
            param[0] -= s
    return args

def cluster_peaks(peaks, msep):
    """Cluster peaks by chemical shift.
    
    Parameters:
    peaks -- array of peaks' chemical shifts
    msep -- minimum separation of clusters in ppm
    """
    distances = scipy.spatial.distance.cdist(peaks, peaks)
    n_clusts, cIDs = scipy.sparse.csgraph.connected_components(
        (distances <= msep).astype(int), 
        directed=False
    )
    return n_clusts, cIDs

def to_stream(dic, data):
    """Prepare data for NMRpipe.
    Adapted from nmrglue.fileio.pipe
    """
    if data.dtype == "complex64":
        data = ng.fileio.pipe.append_data(data)
    data = ng.fileio.pipe.unshape_data(data)

    # create the fdata array
    fdata = ng.fileio.pipe.dic2fdata(dic)

    # write to pipe
    if data.dtype != 'float32':
        raise TypeError('data.dtype is not float32')
    if fdata.dtype != 'float32':
        raise TypeError('fdata.dtype is not float32')

    datastream = fdata.tobytes() + data.tobytes()
    return datastream

# def pipe_process(expt, fid):
#     """Process Bruker FID using NMRPipe and load using nmrglue."""
#     pipe_output = subprocess.run(
#          ["csh", f"{SCDIR}/pipe/proc_{expt}.com", fid],
#         stdout=subprocess.PIPE)
#     return ng.fileio.pipe.read(pipe_output.stdout)
def pipe_process(expt, fid, fixed_len=None):
    """
    Process Bruker FID using nmrglue with a fixed zero-fill length.
    
    Parameters
    ----------
    expt : str
        NMR experiment type ('1H', '13C', etc.)
    fid : str
        Path to Bruker FID folder
    fixed_len : int
        Target zero-filled length for all spectra
    """

    # Read raw Bruker data
    dic_raw, data_raw = ng.bruker.read(fid)
    
    # Create udic from RAW data (gets correct PPM parameters)
    udic = ng.bruker.guess_udic(dic_raw, data_raw)

    # Remove digital filter
    data = ng.bruker.remove_digital_filter(dic_raw, data_raw)
    
    # Amplitude scaling
    amp_scaling = 3.05176e-02 if expt == '13C' else 6.10352e-02
    data = data * amp_scaling
    
    # Line broadening
    lb_hz = 1.0
    sw_h = dic_raw['acqus']['SW_h']
    lb_param = lb_hz / sw_h
    data = ng.proc_base.em(data, lb=lb_param)
    
    # Zero filling (fixed length)
    if fixed_len is None:
        fixed_len = 262144 if expt == '13C' else 65536  # adjust 1H value as needed
    data = ng.proc_base.zf_size(data, fixed_len)

    print("Data length after zero-filling:", len(data))

    # FFT
    data = ng.proc_base.fft(data)
    # Flip the axis for ppm conversion (caused by uc_from_udic?)
    data = data[::-1]

    # Phase
    data = ng.proc_base.ps(data, p0=0.0, p1=0.0)
    
    # Update size in udic
    udic[0]['size'] = data.shape[-1]
    
    # Convert to NMRPipe format
    dic_pipe = ng.pipe.create_dic(udic)
    
    # PPM scale verification
    uc = ng.fileiobase.uc_from_udic(udic, dim=0)
    ppm = uc.ppm_scale()
    print(f"  PPM range: {ppm.min():.2f} to {ppm.max():.2f}, length: {len(data)}")
    
    return dic_pipe, data



# def pipe_bl(dic, data):
#     """Perform baseline correction using NMRPipe."""
#     pipe_output = subprocess.run(
#         ["csh", f"{SCDIR}/pipe/bl.com"],
#         input=to_stream(dic, data),
#         stdout=subprocess.PIPE)
#     return ng.fileio.pipe.read(pipe_output.stdout)
from pybaselines import Baseline

def ng_bl(data, method="auto", poly_order=3):
    """
    Baseline correction replacement for nmrPipe POLY -auto.
    Expects frequency-domain data.
    Safe to run before or after PS.
    """
    spec = np.real(data)

    if method in ("auto", "poly"):
        baseline, _ = Baseline().modpoly(
            spec,
            poly_order=poly_order,
            tol=1e-3,     # intentionally loose, POLY -auto–like
            max_iter=15  # prevents peak-foot overfitting
        )
        spec = spec - baseline

    elif method == "ng":
        # Backward-compatible nmrglue baseline
        from nmrglue.process import proc_bl
        spec = proc_bl.baseline_corrector(spec)

    else:
        raise ValueError(f"Unknown baseline method: {method}")

    return spec




def sg_findpeaks(spec, window_size, poly_order, prominence_sg=2, prominence_fid=2, noise_bounds=None, plot=True):
    """Detect peaks using a second-derivative Savitzky-Golay filter.
    
    TODO: calculate SD from noise region, not whole spec

    Parameters:
    vector : the spectrum object to process
    window_size : SG filter window size, must be odd. Larger window size results in greater
            smoothing.
    poly_order : order of SG polynomial.
    prominence_sg : number of standard deviations for peak-finding minimum prominence, SG.
    prominence_fid : number of standard deviations for peak-finding minimum prominence, FT data.
    noise_bounds : two-ple of index bounds of noise region for std calculation.
    plot : whether to plot the SG second derivative with found peaks
    
    Return lists containing indices of found peaks and estimated widths.
    """
    # Compute Savitsky-Golay filter, 2nd derivative
    signal = spec.data
    fd2 = -1*scipy.signal.savgol_filter(spec.data, window_size, poly_order, deriv=2)

    # Calculate noise thresholds
    if noise_bounds is None:
        err_f = prominence_fid*np.std(signal)
        err_fd2 = prominence_sg*np.std(fd2)
    else:
        err_f = prominence_fid*np.std(signal[slice(*noise_bounds)])
        err_fd2 = prominence_sg*np.std(fd2[slice(*noise_bounds)])

    # Find peaks, estimate widths and amplitudes, filter by noise thresholds
    pks, prp = scipy.signal.find_peaks(fd2, distance=3, width=(3, window_size*1.5), prominence=err_fd2)
    pkw = np.array([abs(prp['right_ips'][i] - prp['left_ips'][i]) for i in range(pks.shape[0])])
    amps = np.array(signal[[int(pk) for pk in pks]])
    pks = pks[amps > err_f]
    pkw = pkw[amps > err_f]
    # x = np.array(range(len(signal), 0, -1))
    x = spec.ppm

    # Plot found peaks
    if plot:
        fd0 = scipy.signal.savgol_filter(signal, window_size, poly_order, deriv=0)
        pklabels = list(range(pks.shape[0]))
        spec.plot_with_labels(pks, pklabels, yys=[10*window_size*fd2, fd0], ppm_bounds=(max(x), min(x)))

    return list(pks), list(pkw)

class Stack():
    def __init__(self, path: str, expt: str, acq1: str):
        self.path = path
        self.basename = os.path.basename(path)
        self.expt = expt
        direct = expt.split('_')[0]
        self.isotope = isotopes[direct]
        self.spectra = {}
        self.ppm_bounds = (self.isotope.ppm_max, self.isotope.ppm_min)
        self.refpeaks = self.load_cfg()
        self.acq1 = acq1

        # Processing parameters
        self.p0 = 0.
        self.p1 = 0.
        self.cf = 0.

        # Detect valid FIDs
        self.ordered_fids = self.detect_spectra(self.acq1)

    def load_cfg(self):
        try:
            refpeaks = pd.read_csv(
                f"{self.path}/cfg_{self.isotope.id}.txt", 
                sep='\t', 
                names=['Shift', 'Compound']
            )
        except FileNotFoundError:
            print("Config file not found, skipping.")
            refpeaks = None
        return refpeaks

    def detect_spectra(self, acq1):
        """Returns a list of spectra IDs for the given isotope."""
        print(f"Inferring {self.expt} spectra (this may take a minute)...", end=None)
        iso_spectra = []
        subdirs = [os.path.basename(dir) for dir in glob.iglob(f"{self.path}/*") if os.path.isdir(dir)]
        sorted_fids = sorted([int(dir) for dir in subdirs if str.isdigit(dir)])
        for fid in sorted_fids:
            if fid < int(acq1):
                continue
            try:
                fn = str(fid)
                print(f"acq1 path: {self.path}/{fn}")
                dic, _ = ng.bruker.read(f"{self.path}/{fn}")
                print(dic['acqus']['PULPROG'])
                if match_pulprog(self.expt, dic['acqus']['PULPROG'], fid=fid):
                    iso_spectra.append(fn)
            except OSError:
                pass
            except IndexError:
                # happens when TD=0 or malformed acqus
                print(f"Skipping {self.path}/{fn}: IndexError (probably TD=0 or malformed Bruker directory)")
                continue
        print(f"\tFound {len(iso_spectra)} spectra for isotope {self.isotope.id}.")
        print("Done.")
        return iso_spectra

    def read_calibration(self):
        print("Reading existing calibration parameters")
        with open(f"{self.path}/calibrate_{self.expt}.txt", "r") as rf:
            rf.readline()
            values = rf.readline().strip('\n').split('\t')
            self.p0 = float(values[0])
            self.p1 = float(values[1])
            self.cf = float(values[2])
            print(f"P0: {self.p0}")
            print(f"P1: {self.p1}")
            print(f"CF: {self.cf}")

    def write_calibration(self):
        print("Writing new calibration parameters")
        with open(f"{self.path}/calibrate_{self.expt}.txt", "w") as wf:
            wf.write("p0\tp1\tcf\n")
            wf.write(f"{self.p0}\t{self.p1}\t{self.cf}\n")

    def calibrate(self, overwrite=False):
        """Get process parameters using NMRglue and write NMRpipe script

        Parameters:
        acq1 -- FID of first spectrum in experiment, timepoint anchor

        Returns: ppm correction to calibrate the x axis
        """
        # Get timestamp anchor
        acq1_dic, _ = ng.bruker.read(f"{self.path}/{self.acq1}")
        self.timestamp = datetime.datetime.fromtimestamp(acq1_dic['acqus']['DATE'])

        if not overwrite and os.path.isfile(f"{self.path}/calibrate_{self.expt}.txt"):
            self.read_calibration()
            return

        # Process initial FID and calibrate phase and ppm shift
        spec_id = self.ordered_fids[0]
        fid = Spectrum(spec_id, self)
        fid.process(manual_ps=True, overwrite=overwrite)

        # In Stack.calibrate(), before CalibrateWindow:
        # print(f"\n{'='*60}")
        # print(f"PPM BEFORE CalibrateWindow:")
        # print(f"  fid.ppm range: {fid.ppm.min():.2f} to {fid.ppm.max():.2f}")
        # print(f"  fid.ppm[0]: {fid.ppm[0]:.2f}")
        # print(f"{'='*60}\n")

        # plt.figure()
        # plt.plot(fid.ppm, fid.data)
        # plt.title(f"Initial spectrum for calibration test")
        # plt.show()

        CalibrateWindow(fid.ppm, fid.data, self)

        # print(f"\n{'='*60}")
        # print(f"PPM AFTER CalibrateWindow:")
        # print(f"  fid.ppm range: {fid.ppm.min():.2f} to {fid.ppm.max():.2f}")
        # print(f"  fid.ppm[0]: {fid.ppm[0]:.2f}")
        # print(f"{'='*60}\n")

        fid.calibrate(update_ft=True)
        self.spectra[spec_id] = fid
        self.write_calibration()

    def process_fids(self, overwrite=False, auto_bl=True, manual_bl=False, 
                     manual_ps=False):
        for spec_id in self.ordered_fids:
            if spec_id in self.spectra:
                spec = self.spectra[spec_id]
                if spec.is_processed:
                    continue
            else:
                spec = Spectrum(spec_id, self)
            try:
                spec.process(manual_ps=manual_ps, overwrite=overwrite, auto_bl=auto_bl, 
                                    manual_bl=manual_bl)
                self.spectra[spec_id] = spec
            except Exception as err:
                print(f"There was an error during the processing of trace {spec_id}:")
                print(Exception, err, '\n')
                print(traceback.format_exc())
                print("\nAborting process routine.")
                return

    def peakfit_fids(self, overwrite=False, method="ng_pick", **kwargs):
        print("Beginning peak fitting procedure for the stack.")
        for spec_id in self.ordered_fids:
            spec = self.spectra[spec_id]
            if spec.is_peakfit:
                print(f"Skipping fid {spec_id}: it is already peak-fit.")
                continue
            if not overwrite and os.path.isfile(f"{spec.path}/peaklist.txt"):
                print(f"Loading existing peaklist for fid {spec_id}.")
                spec.load_peaklist()
                continue
            try:
                spec.peak_fit(peak_method=method, **kwargs)
            except Exception as err:
                print(f"There was an error during the peak fitting of fid {spec_id}:")
                print(Exception, err, '\n')
                print(traceback.format_exc())
                print("\nAborting peak fit routine.")
                return
            if not prompt_continue():
                return
            
    def ridgetrace_fids(self, **kwargs):
        print("Beginning ridge tracing procedure for the stack.")
        for spec_id in self.ordered_fids:
            spec = self.spectra[spec_id]
            if spec.is_ridgetraced:
                print(f"Skipping fid {spec_id}: it is already peak-fit.")
                continue
            try:
                spec.ridge_trace(**kwargs)
            except Exception as err:
                print(f"There was an error during the ridge tracing of fid {spec_id}:")
                print(Exception, err, '\n')
                print(traceback.format_exc())
                print("\nAborting ridge tracing routine.")
                return

    def write_stack(self, outdir=None, suffix=None, from_ridges=False):
        """Write processed stack to Excel."""
        if suffix is None:
            suffix = ""
        else:
            suffix = "_" + suffix
        spectra = []
        vectors = []
        timestamps = []
        areas = []
        areas_index = []
        for spec_id in self.ordered_fids:
            spec = self.spectra[spec_id]
            if spec.is_processed:
                spectra.append(spec)
                vectors.append(spec.data)
                timestamps.append(spec.time_elapsed)
            if from_ridges: 
                if spec.is_ridgetraced:
                    areas.append(spec.ridges)
                    areas_index.append(spec.time_elapsed)
            else:
                if spec.is_peakfit:
                    areas.append(spec.cpd_areas)
                    areas_index.append(spec.time_elapsed)
        print("in write_stack")
        lengths = [len(v) for v in vectors]
        print(set(lengths))  # see all unique lengths
        stack_array = np.array(vectors)
        df = pd.DataFrame(stack_array, columns=spectra[0].ppm, index=timestamps)
        print(f"\tCollected {len(spectra)} processed of {len(self.ordered_fids)} total spectra.")
        print(f"\tCollected {len(areas)} peak-fit of {len(self.ordered_fids)} total spectra.")

        if outdir is None:
            savepath = f"{self.path}/{self.basename}_{self.expt}{suffix}.xlsx"
        else:
            os.makedirs(outdir, exist_ok=True)
            savepath = os.path.join(outdir, f"{self.basename}_{self.expt}{suffix}.xlsx")
        try:
            writer = pd.ExcelWriter(savepath, engine='xlsxwriter')
            print(f"\tOpened {savepath}.")
        except FileNotFoundError:
            print(f"\tCould not save {savepath}: directory does not exist.")
            return
        
        print("\tWriting processed spectra...")
        df = df.T
        df.to_excel(writer, sheet_name='trace', startrow=1)
        ws = writer.sheets['trace']
        ws.write(0, 0, 'trace')
        ws.write(1, 0, 'ppm')
        for i, item in enumerate(self.ordered_fids):
            ws.write(0, i + 1, item)

        if self.refpeaks is not None:
            print("Writing reference shifts...")
            self.refpeaks.to_excel(writer, sheet_name='cfg', index=False, header=False)
        else:
            print("No config loaded, will not write reference shifts.")

        if len(areas) > 0:
            print("Writing peak table...")
            df = pd.DataFrame(areas, index=areas_index)
            df.to_excel(writer, sheet_name='area', index_label="Time")
            ws = writer.sheets['area']
            ws.write(0, 0, 'Time')
        else:
            print("No peak-fit spectra, will not right integrals table.")

        print("Saving file...")
        # writer.save()
        writer.close()
        print(f"Done. Written to {savepath}.")

class PeakList():
    def __init__(self, shifts, peakw, amps, cIDs, assignments=dict()):
        self.shifts = shifts
        self.peakw = peakw
        self.amps = amps
        self.cIDs = cIDs
        self.assignments = assignments

    def filter_peaks(self, mask):
        """Filter peak parameters using a boolean mask.
        
        Parameters:
        mask : the boolean mask with which to filter the peaks. Each position should be
            True to include the peak or False to discard it."""
        self.shifts = self.shifts[mask]
        self.peakw = self.peakw[mask]
        self.amps = self.amps[mask]
        self.cIDs = self.cIDs[mask]

    def ng_shifts(self):
        """Get shifts array formatted for nmrglue"""
        return np.array([(s,) for s in self.shifts])

class Spectrum():
    def __init__(self, id: str, parent: Stack):
        self.id = id
        self.parent = parent
        self.path = f"{self.parent.path}/{self.id}"
        self.is_processed = False
        self.is_peakfit = False
        self.is_ridgetraced = False

    def process(self, manual_ps=False, overwrite=True, auto_bl=True, manual_bl=False):
        """Process 1D-NMR fid.
        Includes LB, ZF, FT, PS, BL, and PPM shift calibration.

        Parameters:
        manual_ps (False) : Interactively phase-correct and update the phase parameters 
            for the parent stack.
        overwrite (True) : True will force overwrite of existing FT-NMR spectra. False 
            will load an existing FT-NMR spectrum if exists, and skip processing.
        auto_bl (True) : Perform automatic baseline correction using NMRPipe.
        manual_bl (False) : Perform interactive baseline correction.
        """
        print(f"Processing fid {self.id}.")
        print("Loading metadata...")
        dic, _ = ng.bruker.read(self.path)
        n_scans = dic['acqus']['NS']
        self.timestamp = datetime.datetime.fromtimestamp(dic['acqus']['DATE'])
        self.time_elapsed = (self.timestamp - self.parent.timestamp).total_seconds()/3600.

        if os.path.exists(f"{self.path}/ft") and not overwrite:
            self.dic, self.data = ng.fileio.pipe.read(f"{self.path}/ft")
            self.data = self.data.real  # ★ ADD THIS ★
            self.uc = ng.pipe.make_uc(self.dic, self.data, dim=0)
            self.ppm = self.uc.ppm_scale()
            self.is_processed = True
            return

        print("Processing spectrum with NMRPipe...")
        # FIXED_LEN = 294282
        self.dic, self.data = pipe_process(self.parent.expt, f'{self.path}')  # Convert, LB, ZF, FT
        # Added lines here to correct ppm shift issue
        self.uc = ng.pipe.make_uc(self.dic, self.data, dim=0)
        self.ppm = self.uc.ppm_scale()
        print(f"PPM range after pipe_process: {self.ppm.min():.2f} to {self.ppm.max():.2f}")

        print("Performing phase correction...")
        self.dic, self.data = ng.process.pipe_proc.ps(self.dic, self.data, p0=self.parent.p0, p1=self.parent.p1)
        # if sum(self.data) < 0:
        #     self.dic, self.data = ng.process.pipe_proc.ps(self.dic, self.data, p0=180, p1=0) # Flip if below axis
        if manual_ps:
            p0, p1 = ng.process.proc_autophase.manual_ps(self.data, notebook=False)
            self.dic, self.data = ng.process.pipe_proc.ps(self.dic, self.data, p0=p0, p1=p1)
            self.parent.p0 += p0
            self.parent.p1 += p1

            # TEMP DEBUG PLOT
            # uc = ng.pipe.make_uc(self.dic, self.data, dim=0)
            # ppm = uc.ppm_scale()
            # plt.plot(ppm, self.data.real)
            # plt.title("Post-manual phase (pre-BL, pre-calibration)")
            # plt.show()

        # ADDED for ng.bruker: Ensure data is real-valued after phase correction
        self.data = self.data.real

        print("Normalizing spectrum intensity...")
        self.dic, self.data = ng.process.pipe_proc.mult(self.dic, self.data, r=n_scans, inv=True) 
        # Removing these lines as they cause ppm to shift higher
        # self.uc = ng.pipe.make_uc(self.dic, self.data, dim=0)
        # self.ppm = self.uc.ppm_scale()

        # still looks normal here
        # plt.plot(self.ppm, self.data)
        # plt.title("After proc mult")
        # plt.show()
        # print("Automatic and manual baseline correction parameters:")
        # print(auto_bl, manual_bl)
        if auto_bl:
            # Ensure real float32 for NMRPipe
            self.data = np.real(self.data).astype(np.float32)

            print("Performing baseline correction using NMRPipe...")
            # self.dic, self.data = pipe_bl(self.dic, self.data)
            # Alternate method using nmrglue
            # self.data = ng_bl(self.data, method="median")
            self.data = ng_bl(self.data, method="auto", poly_order=3)
        if manual_bl:
            print("Performing interactive baseline correction...")
            nodes = np.arange(min(self.ppm), max(self.ppm), 0.4)
            baseline_corrector = Baseline(self.ppm, self.data, nodes, 11, nr=0.1)
            plt.show()
            bl = baseline_corrector.get_baseline()
            self.data = np.float32(self.data) - np.float32(bl)

        # plt.plot(self.ppm, self.data)
        # plt.title("After baseline correction")
        # plt.show()

        print("Calibrating PPM shift...")
        # Spectrum already wrong here
        # TEMP DEBUG PLOT
        # uc = ng.pipe.make_uc(self.dic, self.data, dim=0)
        # ppm = uc.ppm_scale()
        # plt.plot(self.ppm, self.data)
        # plt.title("Before calibrate")
        # plt.show()
        
        self.calibrate()
        
        # TEMP DEBUG PLOT
        # uc = ng.pipe.make_uc(self.dic, self.data, dim=0)
        # ppm = uc.ppm_scale()
        # plt.plot(self.ppm, self.data)
        # plt.title("After calibrate")
        # plt.show()

        print("Writing processed FT spectrum...")
        ng.fileio.pipe.write(f"{self.path}/ft", self.dic, self.data, overwrite=True)
        self.is_processed = True
        print(f"Done processing fid {self.id}.")

    def calibrate(self, update_ft=False):
        # if self.parent.cf != 0:
        #     print("WARNING: cf ignored; ppm axis already calibrated")
        # return
        if self.parent.cf == 0:
            return
        ppm_range = max(self.ppm) - min(self.ppm)
        np = self.data.shape[0]
        shift = self.parent.cf * (np / ppm_range)
        self.dic, self.data = ng.process.pipe_proc.ls(self.dic, self.data, shift, sw=False)
        self.uc = ng.pipe.make_uc(self.dic, self.data, dim=0)
        self.ppm = self.uc.ppm_scale()
        if update_ft:
            ng.fileio.pipe.write(f"{self.path}/ft", self.dic, self.data, overwrite=True)

    def plot(self, yys=[], ppm_bounds=None, show=True, ax=None, **kwargs):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if ppm_bounds == None:
            ppm_bounds = self.parent.ppm_bounds
        ax.plot(self.ppm, self.data, lw=0.5, **kwargs)
        for yy in yys:
            ax.plot(self.ppm, yy, lw=0.5)
        ax.set_xlim(*ppm_bounds)
        if show:
            plt.show()

    def plot_with_labels(self, label_x, labels, yys=[], ppm_bounds=None, label_y=None, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot(yys=yys, ppm_bounds=ppm_bounds, show=False, ax=ax)
        label_ppm = self.ppm[label_x]
        if label_y is None:
            label_y = self.data[label_x]
        ax.plot(label_ppm, label_y, ls='', marker='*', **kwargs)
        for ci, xi, yi in zip(labels, label_ppm, label_y):
            ax.annotate(ci, (xi,yi), textcoords='offset points', xytext=(0,10), ha='center')
        plt.show()

    def get_sigmax(self, center, window_size):
        ppmax = center + window_size/2.
        ppmin = center - window_size/2.
        sigmax = np.max(self.data[(self.ppm > ppmin) & (self.ppm < ppmax)])
        return sigmax

    def plot_window(self, center, window_size, sigmax=None, r_sigmin=0.05):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if sigmax is None:
            sigmax = self.get_sigmax(center, window_size)
        sigmin = -1*r_sigmin*sigmax
        ax.plot(self.ppm, self.data, lw=1., color='k')
        ppmax = center + window_size/2.
        ppmin = center - window_size/2.
        ax.set_xlim((ppmax, ppmin))
        ax.set_ylim((sigmin, sigmax))
        ax.xaxis.set_tick_params(width=1.)
        plt.setp(ax.spines.values(), linewidth=1.)
        ax.get_yaxis().set_ticks([])
        xticks = [round(tk, 1) for tk in ax.get_xticks()]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontname='Arial', size=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.show()

    def ppm_to_float_span(self, xx):
        """Convert a distance in PPM to a distance in points (float representation)."""
        return abs(self.uc.f(0, unit='ppm') - self.uc.f(xx, unit='ppm')) # convert PPM scale to index
    
    def ppm_to_int_span(self, xx):
        """Convert a distance in PPM to a distance in points (integer representation)."""
        return int(self.ppm_to_float_span(xx)) # convert PPM scale to index # convert PPM scale to index
    
    def ppm_to_float(self, xx):
        """Convert a chemical shift in PPM to float representation."""
        return self.uc.f(xx, unit='ppm')
    
    def ppm_to_int(self, xx):
        return int(self.ppm_to_float(xx))
    
    def float_to_ppm(self, yy): 
        """Convert a chemical shift in float or integer representation to PPM."""
        return self.uc.ppm(yy) # convert index to PPM shift
    
    def float_to_ppm_span(self, yy):
        """Convert a distance in points (float or integer representation) to PPM."""
        return abs(self.uc.ppm(0) - self.uc.ppm(yy))

    def peak_pick_sg(self, plot=True):
        """Use Savitzky-Golay method to detect both fine and coarse peaks.
        
        TODO: tune for 13C
        
        Parameters:
        plot : whether to plot the SG second derivative with found peaks
        """
        if self.parent.isotope.id == '1H':
            noise_bounds = (self.ppm_to_int(12), self.ppm_to_int(8))
            coarse, cpeakw = sg_findpeaks(self, 101, 2, prominence_sg=5, prominence_fid=5, noise_bounds=noise_bounds, plot=plot)
            fine, fpeakw = sg_findpeaks(self, 11, 2, prominence_sg=8, prominence_fid=12, noise_bounds=noise_bounds, plot=plot)
        # elif isotope == '13C': # TODO: OPTIMIZE FOR 13C
        #     nr = (int(uc.f(160, unit="ppm")), int(uc.f(130, unit="ppm")))
        #     coarse_peaks = list(sg_peakpick(data, 31, 3, r=2, nr=nr))
        #     fine_peaks = list(sg_peakpick(data, 11, 2, r=5, nr=nr))
        all_peaks = fine + coarse
        all_widths = fpeakw + cpeakw
        shifts = np.array(all_peaks)
        peakw = np.array([(pkw,) for pkw in all_widths])
        amps = np.array([self.data.real[a] for a in fine] + [0.5*self.data.real[a] for a in coarse])
        _, cIDs = cluster_peaks([(sh,) for sh in shifts], self.ppm_to_float_span(self.parent.isotope.fit_cluster_msep))
        cIDs = np.array(cIDs)
        return PeakList(shifts, peakw, amps, cIDs)

    def peaks_from_reference(self):
        """Set peaks using reference shifts from cfg_ISOTOPE.txt."""
        assignments = collections.defaultdict(list)
        shifts = []
        for cpshift, cpd in self.parent.refpeaks.itertuples(index=False, name=None):
            cpshift = int(self.ppm_to_int(cpshift))
            assignments[cpd].append(cpshift)
            shifts.append(cpshift)
        shifts = np.array(shifts)
        peakw = np.array([(self.ppm_to_float_span(0.025),) for _ in shifts])
        amps = np.array([self.data.real[sh] for sh in shifts])
        _, cIDs = cluster_peaks(self.peaklist.ng_shifts(), self.ppm_to_float_span(self.parent.isotope.fit_cluster_msep))
        cIDs = np.array(cIDs)
        return PeakList(shifts, peakw, amps, cIDs, assignments=assignments)

    def peak_pick_ng(self, r=6):
        """Pick peaks using the nmrglue.analysis.peakpick.pick function.
        
        Parameters:
        r : the number of noise standard deviations at which to set the peak detection
            threshold.
        
        Other parameters are defined in isotopes.json.
        """
        msep = self.parent.isotope.peak_pick_msep
        pthres = r*rmse(self.data[(self.ppm <= 160) & (self.ppm >= 130)])
        shifts_tup, cIDs, peakw, amps = ng.analysis.peakpick.pick( # Find peaks
            self.data, pthres=pthres, msep=(self.ppm_to_int_span(msep),),
            algorithm='thres-fast', est_params=True, lineshapes=['l'],
            cluster=True, c_ndil=self.ppm_to_int_span(0.3), table=False
        )
        peaklist = PeakList(
            np.array([s[0] for s in shifts_tup]), # shifts
            np.array(peakw),
            np.array(amps),
            np.array(cIDs)
        )
        return peaklist

    def assign_peaks(self, shifts, cIDs):
        """Assign peaks to clusters using picked peaks, reference shifts, and isotope assignment params."""
        # Prepare arrays and constants for use within the loop
        isotope = self.parent.isotope
        refshifts = np.array([self.ppm_to_float(v) for v in self.parent.refpeaks["Shift"].to_numpy()])
        assignments = collections.defaultdict(list) # dict to store compound assignments
        rev_assignments = dict() # for UI-select
        display_window = self.ppm_to_float_span(isotope.assignment_display_window)
        cluster_tolerance = self.ppm_to_float_span(isotope.assignment_cluster_msep)

        # Assign compound to peaks by proximity to reference shifts, handling collisions
        # Iterate: cluster ID
        for cid in np.unique(cIDs):
            subpeaks = shifts[cIDs == cid] # detected peaks w/in cluster
            clb = subpeaks.min() - cluster_tolerance # calculate cluster bounds
            cub = subpeaks.max() + cluster_tolerance

            # Determine which reference peaks fall within cluster bounds
            close_refpks = [i for i, xx in enumerate(refshifts) if xx > clb and xx < cub]

            # Assign peaks to compounds; skip clusters with no nearby reference shifts
            if len(close_refpks) == 1:
                # If only one matched ref. peak, assign all cluster sub-peaks to that compound
                cpd = self.parent.refpeaks.at[close_refpks[0], 'Compound']
                assignments[cpd].extend(subpeaks)
                rev_assignments.update({pk: cpd for pk in subpeaks})
            elif len(close_refpks) > 1: # Collision!
                # If multiple ref. peaks, manually assign sub-peaks in plot window
                colliding_refs = self.parent.refpeaks.loc[close_refpks, :]
                print(f"Cluster {cid} assigned to {len(close_refpks)} conflicting reference peaks.")
                print("In the plot, find the contributions from each peak using the reference shifts listed.")
                ppm_bounds = (
                    self.float_to_ppm(min(subpeaks) - display_window), 
                    self.float_to_ppm(max(subpeaks) + display_window)
                )
                to_update = {}
                ConflictWindow(self.ppm, self.data, ppm_bounds, cid, subpeaks, colliding_refs, to_update)
                for cpd, pks in to_update.items():
                    assignments[cpd].extend(pks)
                    rev_assignments.update({pk: cpd for pk in pks})

        return assignments, rev_assignments

    def define_curveshape(self, cv):
        """Define curve fitting parameters for curveshapes.
        
        Parameters:
        cv : string representing curveshape. Supported shapes:
            - l : Lorentzian
            - g : Gaussian
            - v : Voigt (Lorentzian/Gaussian convolution)
            - gb : Gubmel function (bell shape with skew)
            - sl : Split Lorentzian
        """
        isotope = self.parent.isotope
        peaklist = self.peaklist
        ppm_delta = self.ppm_to_float_span(isotope.fit_ppm_delta)
        fwhm_max = self.ppm_to_float_span(isotope.fit_fwhm_max)
        if cv == 'sl':
            # Special case for split Lorentzian
            amps = []
            amp_bounds = []
            for x in self.shifts:
                delp = 0.209*0.6
                ppm_range = (self.ppm < (self.float_to_ppm(x) + delp)) \
                            & (self.ppm > (self.float_to_ppm(x) - delp))
                maxb = self.data[ppm_range].max()
                amps.append(maxb)
                amp_bounds.append((0.05, 1.2*maxb))
            amps = np.array(amps)
            amp_bounds = np.array(amp_bounds)
        else:
            # amps = np.array([self.data.real[int(a)] for a in self.shifts], dtype=np.float64)
            amps = self.peaklist.amps
            amp_bounds = np.array([(0.05, 1.2*a) for a in amps])

        # Define start parameters and bounds for each peak
        if cv == 'v':
            cparams = np.array([((a, b[0], b[0]),) for a, b in zip(peaklist.shifts, peaklist.peakw)])
            cbounds = []
            for par in cparams:
                cbounds.append(((
                    (par[0][0] - ppm_delta, par[0][0] + ppm_delta),    # shift bounds
                    (0., self.ppm_to_float_span(fwhm_max)), # gauss bounds
                    (0., self.ppm_to_float_span(fwhm_max)), # lorentz bounds
                ),))
        elif cv == 'l' or cv == 'g':
            # Lorentzian function
            cparams = np.array([((a, b[0]),) for a, b in zip(peaklist.shifts, peaklist.peakw)])
            cbounds = []
            for par in cparams:
                cbounds.append(((
                    (par[0][0] - ppm_delta, par[0][0] + ppm_delta),    # shift bounds
                    (0., fwhm_max), # lorentz or gauss bounds
                ),))
        elif cv == 'gb':
            # Custom Gumbel function
            cparams = np.array([((a, -1*b[0]),) for a, b in zip(peaklist.shifts, peaklist.peakw)])
            cbounds = []
            for par in cparams:
                cbounds.append(((
                    (par[0][0] - ppm_delta, par[0][0] + ppm_delta),    # shift bounds
                    (-1*fwhm_max, 0.), # beta bounds
                ),))
            cv = gumbel_hb()
        elif cv == 'sl':
            # Custom split lorentzian function. Split peaks have identical parameters.
            cparams = np.array([((a, b[0], self.ppm_to_float_span(0.209)),) for a, b in zip(peaklist.shifts, peaklist.peakw)])
            cbounds = []
            for par in cparams:
                cbounds.append(((
                        (par[0][0] - ppm_delta, par[0][0] + ppm_delta),    # shift bounds
                        (0., self.ppm_to_float_span(fwhm_max)), # lorentz bounds
                        (self.ppm_to_float_span(0.208), self.ppm_to_float_span(0.210)),   # J-constant bounds
                    ),))
            cv = split_lorentz_fwhm()

        return cv, cparams, cbounds, amps, amp_bounds

    def curve_fit(self, cv='l'):
        """Perform curve fitting in windows."""
        cv, cparams, cbounds, amps, amp_bounds = self.define_curveshape(cv)
        self.cv = cv
        shifts_tup = self.peaklist.ng_shifts()
        isotope = self.parent.isotope
        cluster_sep = self.ppm_to_float_span(isotope.fit_cluster_msep)
        fit_reach = self.ppm_to_float_span(isotope.fit_reach)
        cbounds = np.array(cbounds)
        rIDs = np.array([1 for _ in shifts_tup])

        # Prepare arrays and constants for use within the loop
        curve_params = [cparams, amps, cbounds, amp_bounds, shifts_tup, rIDs]
        fit_params = np.zeros_like(cparams)      # array for fit result, curve shift + FWHM
        fit_amps = np.zeros_like(amps)      # array for fit result, curve amplitudes

        # Calculate curve fitting windows and iterate for curve fit
        n_fitwd, fitwd = cluster_peaks(shifts_tup, cluster_sep)
        for wid in range(n_fitwd):
            mask = (fitwd == wid)
            fit_args = [pset[mask] for pset in curve_params]
            
            # slice region from +/- <cluster_reach> for curve-fitting
            lb = min(self.peaklist.shifts[mask]) - cluster_sep # 0.5*cluster_sep
            ub = max(self.peaklist.shifts[mask]) + cluster_sep # 0.5*cluster_sep
            _, ndata = ng.process.pipe_proc.ext(self.dic, self.data, x1=lb, xn=ub)

            # Shift peak parameters into slice and perform curve fit
            fit_args[4] = shift_params(lb, fit_args[4])[0]  # Shift ppm shift
            fit_args[0] = shift_params(lb, *fit_args[0])    # Shift VP params
            fit_args[2] = shift_params(lb, *fit_args[2])    # Shift VP param bounds
            maxfev = 200000
            nparams, namps, _ = ng.analysis.linesh.fit_spectrum(
                ndata, 
                [cv],
                *fit_args,
                self.ppm_to_int_span(fit_reach), 
                False, 
                verb=False,
                maxfev=200000,
            )
            nparams = shift_params(-1*lb, *np.array(nparams)) # rescale result to full PPM axis

            # Deposit curve-fit parameters
            fit_params[mask] = np.array(nparams)
            fit_amps[mask] = np.array(namps)

        self.fit_params = fit_params
        self.fit_amps = fit_amps

    def pack_peaks(self):
        """Pack parameters into Peak object for manual assignment."""
        rev_assignments = {}
        for cpd, pklist in self.peaklist.assignments.items():
            rev_assignments.update({pk: cpd for pk in pklist})
        peak_curves = []
        for sh, pars, amp in zip(self.peaklist.shifts, self.fit_params, self.fit_amps):
            shift = self.float_to_ppm(pars[0][0])
            fwhm = self.float_to_ppm_span(pars[0][1])
            cpd = rev_assignments.get(sh, "Unassigned")
            peak_curves.append(Peak(self.parent.isotope, shift, amp, l_fwhm=fwhm, cpd=cpd))
        return peak_curves

    def unpack_peaks(self, ui_peaks):
        """Unpack parameters from Peak object into nmrglue-compatible array structures."""
        shifts = []
        peakw = []
        amps = []
        all_cpds = []
        assignments = collections.defaultdict(list)
        for pk in ui_peaks:
            if pk.lw == 0.0 or pk.y == 0.0:
                continue
            sh = self.ppm_to_int(pk.get_x())
            shifts.append(sh)
            peakw.append((self.ppm_to_float_span(pk.get_lw()),))
            amps.append(pk.y)
            assignments[pk.cpd].append(sh)
            all_cpds.append(pk.cpd)
        _, cIDs = cluster_peaks([(sh,) for sh in shifts], self.parent.isotope.fit_cluster_msep)
        self.peaklist = PeakList(np.array(shifts), np.array(peakw), np.array(amps), cIDs, assignments=assignments)
        self.fit_params = np.array([((a, b[0]),) for a, b in zip(self.peaklist.shifts, self.peaklist.peakw)])
        self.fit_amps = self.peaklist.amps
        self.all_cpds = np.array(all_cpds)

    def refine_curves(self):
        print("Initial peak fit complete. Packing parameters for manual assignment...")
        peak_curves = self.pack_peaks()
        print("Begin manual assignment.")
        w = PeakFitWindow(
            self.parent.isotope, 
            peak_curves, 
            self,
        ) # edits peak_curves in-place
        w.display()
        peak_curves = w.get_peaks()
        print("Unpacking parameters...")
        self.unpack_peaks(peak_curves)
        print("Writing peak parameters...")
        self.write_peaks(peak_curves)

    def simulate_peaks(self, resolution):
        """Simulate curve-fit peaks and integrate."""
        scale = resolution/self.data.shape[0]
        self.sim_ppm = np.linspace(self.ppm[0], self.ppm[-1], num=resolution)
        
        self.sim_params = scale*self.fit_params
        sim_peaks = []
        sim_areas = []
        
        for params, amps in zip(self.fit_params, self.fit_amps):
            sim_peak = ng.analysis.linesh.sim_NDregion( # Simulate spectrum in high-resolution
                (resolution,), 
                [self.cv], 
                [scale*params], 
                [amps],
            )
            sim_peaks.append(sim_peak)
            sim_areas.append(-1*np.trapz(sim_peak, x=self.sim_ppm))

        self.sim_peaks = np.array(sim_peaks)
        self.sim_areas = np.array(sim_areas)
        self.sim_curve = np.sum(self.sim_peaks, axis=0)

        self.cpd_areas = {}
        for cpd, area in zip(self.all_cpds, self.sim_areas):
            self.cpd_areas.update({
                cpd: self.cpd_areas.get(cpd, 0.) + area
            })
    
    def write_peaks(self, peak_curves):
        """Write peak list to FID directory"""
        peak_records = [pk.to_record() for pk in peak_curves]
        df = pd.DataFrame(peak_records)
        df.to_csv(f"{self.path}/peaklist.txt", sep="\t")

    def calculate_residual(self):
        """Calculate residual from simulated spectrum."""
        sim_native = ng.analysis.linesh.sim_NDregion( # Simulate spectrum in native resolution
            (self.data.shape[0],), 
            [self.cv], 
            self.fit_params, 
            self.fit_amps,
        )
        self.residuals = self.data - sim_native

    def plot_fit_curves(self):
        """Plot simulated spectrum and constituent curves."""
        fig, ax = plt.subplots()
        self.plot(ax=ax, zorder=2, show=False)
        cpd_sim_specs = {}
        # cpd_colors = {cpd: next(ax._get_lines.prop_cycler)['color'] for cpd in np.unique(self.all_cpds)}
        color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        cpd_colors = {cpd: next(color_cycle) for cpd in np.unique(self.all_cpds)}

        ppm_shifts = [self.float_to_ppm(sh) for sh in self.peaklist.shifts]

        # Plot simulated peaks with annotated summits
        for cpd, sim_peak, shift, amp in zip(self.all_cpds, self.sim_peaks, ppm_shifts, self.fit_amps):
            color = cpd_colors[cpd]
            ax.plot(self.sim_ppm, sim_peak, lw=0.5, color=color, zorder=4)
            ax.plot(shift, amp, ls='', marker='*', color=color, zorder=5)
            ax.annotate(cpd, (shift,amp), textcoords='offset points', xytext=(0,10), ha='center', zorder=6)
            cpd_spec = cpd_sim_specs.get(cpd, np.zeros_like(self.sim_ppm))
            cpd_sim_specs.update({
                cpd: cpd_spec + sim_peak
            })

        # Plot sum curve for each compound
        for cpd, sim_spec in cpd_sim_specs.items():
            color = cpd_colors[cpd]
            ax.plot(self.sim_ppm, sim_spec, lw=1, color=color, zorder=4)

        # Plot total simulated signal + residual and show plot
        ax.plot(self.sim_ppm, self.sim_curve, lw=0.5, color='k', zorder=3)
        ax.plot(self.ppm, self.residuals, lw=0.5, color='gray', zorder=1)
        ax.set_xlim(self.parent.ppm_bounds)
        plt.show()

    def peak_fit(self, r=8, sep=0.005, peak_method='ng_pick', plot=True):
        """Peak-pick NMR spectrum.
        Includes peak-picking, peak assignment, and integration functionality. The
        reference peaks to be fit should be specified in cfg_{isotope}.txt and
        provided in the directory of the run to be processed.

        Parameters:
        ppm -- chemical shift axis of the data, in ppm
        data -- signal axis of the data
        r -- RMSE noise threshold for peak detection (default: 6)
        plot -- True (default) to plot fit peaks with cluster labels over the data.
        vb -- True for verbose mode.

        Returns a dictionary mapping compounds to their peak areas, a curve
        summing all of the fit peaks, and a list of lists acting as a table of peak data.
        """
        print(f"Beginning peak-fitting routine for spectrum {self.id}.")
        plot_bounds = self.parent.ppm_bounds
        # plot_bounds_hz = [uc.hz(uc.f(ele, unit='ppm')) for ele in plot_bounds]

        print(f"Performing peak picking subroutine with method {peak_method}...")
        if peak_method == 'from_cfg':
            self.peaklist = self.peaks_from_reference()
        elif peak_method == 'sg_pick':
            self.peaklist = self.peak_pick_sg()
        elif peak_method == 'ng_pick':
            self.peaklist = self.peak_pick_ng(r=r)

        if len(self.peaklist.shifts) == 0:
            # Exit if no peaks detected
            print("No peaks detected; exit.")
            return {}, []

        if plot:
            self.plot_with_labels(self.peaklist.shifts, self.peaklist.cIDs, ppm_bounds=plot_bounds)
            plt.show()

        if peak_method == 'ng_pick' or peak_method == 'sg_pick':
            print("Performing peak assignment subroutine...")
            assignments, rev_assignments = self.assign_peaks(self.peaklist.shifts, self.peaklist.cIDs)
            self.peaklist.assignments = assignments

        if len(assignments) == 0:
            print("No reference peaks detected; exit.")
            return {}, []

        print("Filtering peaks...")
        all_cpds = []
        used_pks = []
        for cpd, apks in assignments.items():
            if len(apks) > 0:
                all_cpds.append(cpd)
                used_pks.extend(apks)
        all_cpds = np.array(all_cpds)
        self.peaklist.filter_peaks(np.isin(self.peaklist.shifts, used_pks))

        print("Performing curve-fitting subroutine...")
        self.curve_fit()

        if peak_method == 'sg_pick' or peak_method == 'ng_pick':
            print("Performing interactive curve refinement...")
            self.refine_curves()

        print("Performing simulation and integration subroutine...")
        self.simulate_peaks(CURVE_FIT_RESOLUTION)
        self.calculate_residual()
        self.is_peakfit = True
        
        print("Peak-fitting routine complete.")
        self.plot_fit_curves()

    def ridge_trace(self, wr=0.01, plot=True):
        """Simple function to get the heights of reference peaks in a spectrum.
        Takes the maximum signal within window of each peak. Does not functionally
        trace ridge across spectra; cannot handle drifting chemical shifts, eg. due
        to pH changes. Shift must remain within window of reference throughout time
        course.

        Parameters:
        wr -- window radius, distance from reference shift to include in window

        Returns a dictionary containing the amplitude of each peak.
        """
        signals = {}
        shifts = []
        amps = []
        for i, row in self.parent.refpeaks.iterrows():
            shift = float(row["Shift"])
            bounds = (self.ppm_to_int(shift-wr), self.ppm_to_int(shift+wr))
            peak = f"{row['Compound']} {shift}"
            amplitude = max(self.data[bounds[1]:bounds[0]])
            signals[peak] = amplitude
            shifts.append(shift)
            amps.append(amplitude)
        if plot:
            self.plot_with_labels(np.array([self.ppm_to_int(s) for s in shifts]),
                # [str(s) for s in shifts], label_y=amps, ppm_bounds=(200., 0.))
                # [str(s) for s in shifts], label_y=amps, ppm_bounds=(12., 0.))
                [str(s) for s in shifts], label_y=amps, ppm_bounds=(14., -0.5))

        self.ridges = signals
        self.is_ridgetraced = True

    def load_peaklist(self):
        peakdata = pd.read_csv(f"{self.path}/peaklist.txt", sep='\t', index_col=0)
        peakdata['ppm_i'] = peakdata['ppm'].map(self.ppm_to_int)
        _, cIDs = cluster_peaks([(sh,) for sh in peakdata['ppm_i']], self.parent.isotope.fit_cluster_msep)
        assignments = {cpd: seg['ppm'].to_list() for cpd, seg in peakdata.groupby(by='cpd')}
        self.peaklist = PeakList(
            peakdata['ppm_i'].to_numpy(), 
            np.array([(self.ppm_to_float_span(fwhm),) for fwhm in peakdata['fwhm']]), 
            peakdata['amp'].to_numpy(), 
            cIDs, 
            assignments=assignments
        )
        self.cv = 'l'
        self.fit_params = np.array([((a, b[0]),) for a, b in zip(self.peaklist.shifts, self.peaklist.peakw)])
        self.fit_amps = self.peaklist.amps
        self.all_cpds = peakdata['cpd'].to_numpy()
        self.simulate_peaks(CURVE_FIT_RESOLUTION)
        self.is_peakfit = True

def call_main():
    """Parse the command line input and create a Stack object."""
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', metavar='DATAPATH', help='path to dataset')
    parser.add_argument('expt', metavar='EXPT', help='experiment string (1H, 13C, or 1H_13C)')
    parser.add_argument('acq1', metavar='INITIAL', help='index of the initial run FID, for calibration')
    args = parser.parse_args()
    if not os.path.exists(args.datapath):
        print("Not a valid path " + args.datapath)
        return
    supported_exp = ['1H', '13C', '1H_13C']
    if args.expt not in supported_exp:
        print(f"Unsupported experiment {args.expt}. Supported: {', '.join(supported_exp)}")
        return
    if not os.path.exists(f"{args.datapath}/{args.acq1}"):
        print(f"FID {args.acq1} was not found in the datapath. Please choose a valid FID.")
        return
    s = Stack(args.datapath, args.expt, args.acq1)
    s.calibrate()
    s.process_fids()
    s.peakfit_fids()
    s.write_stack()
    
if __name__ == "__main__":
    call_main()
