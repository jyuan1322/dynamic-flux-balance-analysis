# Tutorial for NMR processing pipeline
This tutorial assumes a working knowledge of the command line. The instructions were written for Mac users but could be adapted for Windows systems with small modifications. Each section includes an estimate of how long it will take.  

1. [Preparing to run the pipeline (20 minutes)](TUTORIAL.md#preparing-to-run-the-pipeline-20-minutes)
2. [Processing the NMR runs (15 minutes per run)](TUTORIAL.md#processing-the-nmr-runs-15-minutes-per-run)
3. [Running the dFBA analyses (5 minutes)](TUTORIAL.md#running-the-dfba-analyses-5-minutes)

## Preparing to run the pipeline (20 minutes)

### Installing the dependencies

Make sure all requirements are satisfied in the "Installation" and "Dependencies" sections of [README.md](README.md). A c-shell environment (csh/tcsh) must be activated when installing NMRPipe and working with `process.py`: 
1. Install [NMRPipe](https://www.ibbr.umd.edu/nmrpipe/install.html). On a Mac, the installation process should look something like this:
    ```
    csh
    cd
    mkdir nmr
    cd nmr

    curl -O https://www.ibbr.umd.edu/nmrpipe/install.com
    curl -O https://www.ibbr.umd.edu/nmrpipe/binval.com
    curl -O https://www.ibbr.umd.edu/nmrpipe/NMRPipeX.tZ
    curl -O https://www.ibbr.umd.edu/nmrpipe/s.tZ
    curl -O https://www.ibbr.umd.edu/nmrpipe/dyn.tZ

    chmod a+r  *.tZ *.Z *.tar
    chmod a+rx *.com

    ./install.com
    ```
    Check your `.cshrc` file for the initialize commands.
    ```
    cd
    cat .cshrc
    ```
      The file should contain the following lines:
    ```
    if (-e /path/to/nmr/com/nmrInit.mac11_64.com) then
       source /path/to/nmr/com/nmrInit.mac11_64.com
    endif
    ```
    If the lines are missing or the .cshrc file does not exist, open the file `README_NMRPIPE_USERS` in the `nmr` folder you just created and find the lines that match the ones above. Copy them into .cshrc file. Then log out and log back in again.  

2. Download the repository with `git clone https://github.com/Massachusetts-Host-Microbiome-Center/nmr-cdiff.git` or using the GitHub Desktop app.  
3. Set up the virtual environment. From the `nmr-cdiff` base directory:
    ```
    cd etc
    python3 -m venv env
    source env/bin/activate.csh
    python3 -m pip install -r requirements.txt
    ```
    Now, you have all the dependencies for this repository installed in the virtual environment `env`. Now that the environment is set up, you do not need to run the above code again. However, when you start a session, you must activate the virtual environment as follows:
    ```
    csh
    source nmr-cdiff/etc/env/bin/activate.csh 
    ```       

### Test data

One short unprocessed NMR run and two processed runs are included in the [test data](data/test) folder:
 - [20210519_13CGlc](data/test/20210519_13CGlc) : a short unprocessed glucose run  
 - [20210322_13CLeu](data/test/20210322_13CLeu) : a short processed leucine run  
 - [20220421_13CGlc_standards](data/test/20220421_13CGlc_standards) : a processed dataset of standard solutions supporting the dFBA analyses

We will process the glucose run and simulate dFBA. For convenience, we have included `cfg_13C.txt` and `cfg_1H.txt` in the run directory, which contain reference chemical shifts for the substrate and expected products. For each substrate, these files must be prepared before processing is run.

## Processing the NMR runs (15 minutes per run)
### The Stack object
We will begin by processing the NMR datasets using the [process.py](scripts/process/process.py) script. Navigate to the directory containing the python scripts, and start an interactive python session.  
```
cd nmr-cdiff/scripts/process
python3
```
Now, we will import the Stack class from process.py and create a Stack object for our NMR run. This will serve as the basis for our processing functions. When we create the Stack object, we will need to specify the filepath of the run, as well as the isotope and the spectrum FID to use as the timestamp anchor.
```
from process import Stack
s = Stack("nmr-cdiff/data/test/20210519_13CGlc", "13C", "52")
```
This will create a stack object for our test glucose run, and specify that we want to process the 13C spectra with FID number 52 as our anchor.  

Our `Stack` object `s` has several useful attributes and methods that we will use:  
Attributes:  
* `s.refpeaks` : a pandas dataframe that stores the reference peaks from `cfg_13C.txt`. You can manually load a new config with the line `s.refpeaks = s.load_cfg()`.  
* `s.ordered_fids` : a list of strings, representing the names of the FIDs to be included in the Stack. Set this manually if automatic FID detection fails.  
* `s.spectra` : a dictionary mapping the FIDs' string labels to Spectrum objects. For example, to access FID \#23, use `spec = s.spectra["23"]`. This dictionary is populated by `s.process_fids()`.  

Methods:  
* `s.calibrate()` : a method that finds calibration parameters for processing. This method must be run before `process_fids()`.  
* `s.process_fids()` : this function process all FIDs in a Stack. The processing function `process_fids` has several keyword arguments that affect the function's behavior:  
    * `man_ps` : whether to perform manual phase correction for each spectrum (default: `False`).  
    * `auto_bl` : whether to use NMRPipe's automatic baseline correction (default: `True`, recommended for 13C spectra).  
    * `man_bl` : whether to use the in-house manual baseline correction with draggable nodes and spline fit (default: `False`, recommended for 1H\[13Ced\] spectra).  
    * `overwrite` : whether to force overwrite of existing processed spectra (default: `False`).  
* `s.peakfit_fids()` : a method to perform peak-picking and curve fitting on all spectra in the Stack. In addition to the following keyword arguments, this function can pass any additioal keyword arguments to `spec.peak_fit()`.
    * `overwrite` : whether to force overwrite of existing fit peaks (default: `False`).
    * `method` : the method to use for finding peaks. Supported: `"ng_pick"` (default): use nmrglue's peak-picking algorithm, recommended for 13C; `"sg_pick"`: use in-house Savitsky-Golay filtering algorithm, recommended for 1H\[13Ced\].  
* `s.ridgetrace_fids()` : an alternative to `peakfit_fids`, this method tracks amplitude rather than peak area for all peaks specified in `refpeaks`. 
* `s.write_stack()` : write the processed spectra, reference peaks, and/or compound-designated peak areas to an Excel spreadsheet. This can be thought of as "exporting" the processed Stack.
    * `suffix` : a suffix to add to the end of the filename, which is helpful if different versions are desired.
    * `from_ridges` : whether to populate the "areas" tab with the `ridgetrace_fids` output instead of the `peakfit_fids` output (default: `False`).

### Find the calibration parameters
Let's first find some calibration parameters to make processing go faster. We will need to run our Stack's `calibrate` method.
```
s.calibrate()
```
1. You will first be prompted for phase correction.  
    1. First, find the ppm shift of the **right-most** peak using the cursor on the plot. Next, slide the **pivot** slider to match this value.  
    2. Adjust the **p0** slider until the **right-most** peak is well-phased.  
    3. Adjust the **p1** slider until the **left-most** peak is well-phased.  
    4. Adjust **p0** and **p1** slightly until all peaks are well-phased.  

    Press the "Set Phases" button, and close the window. For this spectrum, **p0=70.3** and **p1=99.9** were appropriate values.  

1. Next, you will be prompted to calibrate the reference shift.
    1. Find a well-separated peak in the reference 13C spectrum of Glucose. The peak at **72.405 ppm** appears suitable in [this reference spectrum from HMDB](https://hmdb.ca/spectra/nmr_one_d/166522). Type this in the field for the reference peak.
    1. Next, find the corresponding peak in the spectrum, which may be slightly shifted from 72.405 ppm. Use the zoom tool to locate the peak and cursor to find the chemical shift at the center of the peak. Enter this experimental chemical shift and click Submit.  

These parameters will be stored to aid with processing each remaining spectrum in the run. Additionally, they will be saved so that if you exit the python session and create a new Stack for this run, the stored parameters will be loaded. To forget these stored parameters, delete calibrate_13C.txt from the run folder.

### Process the 13C spectra
Processing will now begin, and should be rather quick. Run `process_fids`:
```
s.process_fids()
```
Unless you have set `man_ps=True` or `man_bl=True`, this should be entirely automated, and will likely take less than a minute. Use `man_ps=True` to adjust the phase correction of each spectrum individually as described above, recommended if the phase deviates throughout the course of the run. Use `man_bl=True` for spectra where the NMRPipe baseline correction fails, especially for 1H\[13Ced\] spectra. For `man_bl=True`, adjust the nodes for the spline fit by removing or dragging nodes not on the baseline. Note: ensure that the order of the nodes is maintained as you drag them. Also, adding new nodes may cause unexpected behavior as this function has not been thoroughly tested.

### Peak-fit the 13C spectra
Now, we will fit curves to the NMR peaks:
```
s.peakfit_fids()
```

1. The peak-picking algorithm will automatically find peaks in the spectrum. At times, manual peak assignment may be required if peaks are close to each other. If this is the case, a pop-up window will show where each peak is labeled by a numbered index. In the right panel, the conflicting reference shifts will be listed. Make note of which peaks belong to which compounds. In the right panel, list the indices belonging to each compound, separated by spaces. Click Submit.  
1. The program will next perform curve-fitting on the found peaks. If this is taking too long, it may be beneficial to interrupt the processing and try running `peakfit_fids` with the `r` keyword argument set to a higher value (default: 8). This will raise the noise threshold for peak detection. It will often need to be set to a higher level for spectra with very high S/N ratio, where noise at the base of the peak will be detected as peaks at low values of `r`.  
1. After automatic peak picking and fitting is completed for each spectrum, the spectrum will be plotted with the curve-fit and peak assignments overlaid. Advanced users may wish to inspect this spectrum for proper curve-fitting and add, remove, or adjust curves as needed. Use the "Edit Peak" tab to select and modify the parameters of existing peaks, the "Add Peak" tab to add a new peak, and the "Remove Peak" tab to remove the active peak.  

    In the plot window, the real spectrum is drawn in black, the simulated spectrum in blue, and the residuals in gray. The active peak is highlighted in red. The contributions of individual compounds are also plotted in different colors.  

    When selecting peaks or performing actions, it may take a few seconds for the active peak to update in the plot. Please be patient; if it's taking too long, try selecting different peaks in the dropdown to force an update.  

    If you have made modifications to the peaks, press the "Curve Fit" button before closing out to ensure that the curves form an optimal solution set.  
    
1. Once curve fitting is completed for the spectrum, the curve shape parameters will be saved in `peaklist.txt`.
1. You will be prompted to continue to the next spectrum, or exit the peak-fitting routine without risk. Spectra that have already been processed will be skipped on the next pass of `peakfit_fids` and the parameters will be retained.

### Troubleshooting the processing routines
Occasionally, mistakes will be made or adjustments will be necessary in spectra that have already been peak-fit. Thankfully, the object-oriented interface affords the flexibility to make any necessary adjustments.
* To re-do the entire processing routine, create a new `Stack` object for the run. Calibrate the run and use `s.process_fids(overwrite=True)`.  
* To re-do the entire peak-fitting routine, create a new `Stack` object for the run. Run the workflow as usual, but use `s.peakfit_fids(overwrite=True)`.  
* To re-process just one spectrum:
    ```
    spec = s.spectra["23"]      # access the Spectrum object for FID 23
    spec.is_processed = False   # set this flag to False to override the existing processed spectrum
    spec.process()              # process the spectrum again, can pass new kwargs (manual_ps, auto_bl, manual_bl)
    ```  
* To re-do the peak-fitting for just one spectrum (eg. spectrum 23):
    ```
    spec = s.spectra["23"]      # access the Spectrum object for FID 23
    spec.is_peakfit = False     # set this flag to False to override the existing peak-fitting output
    spec.peak_fit()             # peak-fit the spectrum again, can pass new kwargs (r, sep, peak_method)
    ```  
### Writing the processed stack
Once the 13C spectra are processed and peak-fit, all that remains is to export the data into an Excel spreadsheet that can be read by downstream applications.
```
s.write_stack()
```

## Process the 1H spectra
Process the 1H spectra in much the same way.
```
s2 = Stack("nmr-cdiff/data/test/20210519_13CGlc", "1H", "52")
s2.calibrate()
s2.process_fids()
```  
Follow the prompts as before. We used p0=155.8, p1=50.2, reference shift=1.9059, experimental shift=1.9046. If the baseline is not satisfactory, try using `s.process_fids(auto_bl=False, man_bl=True)`.  

The 1H spectra are used to align the time axis based on the production of isocaproate. We look for the isocaproate peak at 0.8642 ppm (0.76 ppm in the 13C-leucine spectra due to peak splitting).
Because we are not looking to quantify estimated concentrations here, we can use a simple function to trace the amplitude of the peak over time.
```
s2.ridgetrace_fids()
```
When we write the stack, we must specify that we wish to use this simple ridge-trace output rather than a peak-fitting output.
```
s2.write_stack(from_ridges=True)
```

### Plotting the runs
A fully processed run can be plotted as a stack, as seen in Fig. 2a,c,e, using the MATLAB function [`plotStacks.m`](scripts/MATLAB/plotStack.m). This can be accomplished with relative ease using the script [`plot_stack.m`](scripts/MATLAB/call/plot_stack.m). Simply add a case for the run and set the variable `met`. You may also need to add a working filepath using the funciton `addpath`. The script is best run in batch mode from the command line:
```
matlab -nodisplay -nodesktop -batch "run('plot_stack.m');exit;"
```

### Additional processed runs
- The run [`20210322_13CLeu`](data/test/20210322_13CLeu) has already been processed for convenience.  
- The run [`20220421_13CGlc_standards`](data/test/20220421_13CGlc_standards) is a processed dataset of standard solutions supporting concentration estimates in the glucose run. The file `Concentrations.xlsx` in this directory contains the concentration of each compound in the processed spectra.

## Running the dFBA analyses (5 minutes)
### Overview of the script
Now that the runs are processed, we will simulate dFBA. The dFBA script will perform the following functions:  
1. Synchronize the NMR runs using the isocaproate trajectory from the 1H spectra.
2. Estimate concentration curves of the substrates and products using standard solutions.
3. Simulate dFBA using the estimated concentration curves as constraints for exchange fluxes.  

### The dFBA config file
Before running the dFBA analyses, we must first define parameters for the dFBA and specify paths to all the datasets that will be used. The structure of the file is as follows:
* `"method"` : the dFBA method to use for simulations (supported: `"fba"`, `"pfba"`, `"loopless"`, or `"fva"`).
* `"model_file"` : path to the metabolic model file to be used for the simulations. Can be an absolute path, or a relative path with respect to dfba.py.

### Run the dFBA analyses
Run the dFBA script, [`dfba.py`](scripts/process/dfba.py):
```
cd nmr-cdiff/scripts/process
python dfba.py dfba_cfg.json
```
The argument is a config file in json format, which specifies the dFBA parameters. This config file is described fully above.

### dFBA output
The dFBA program will display some text output and graphs with the simulation results. Output from tracked reactions and metabolites will be written to `data/fluxes.xlsx`, `data/met_fluxes.xlsx`, and `data/<met>flux_in.txt` and `data/<met>flux_out.txt`. These files support MATLAB plotting functions.

The logistic concentration curves should appear as follows:

<img width="362" alt="dfba1" src="https://user-images.githubusercontent.com/29278926/174334454-8b1242f2-aacd-4e83-b634-4b191be7154d.png">
<img width="1552" alt="dfba2" src="https://user-images.githubusercontent.com/29278926/174334456-e5621b9c-ac7d-4ecc-a965-21d9ed6ea5dc.png">

Our dFBA output was:

<img width="1552" alt="dfba3" src="https://user-images.githubusercontent.com/29278926/174334458-77a59d69-68e0-41fb-bd0a-94de486ebf48.png">

