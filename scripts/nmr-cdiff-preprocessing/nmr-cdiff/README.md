# nmr-cdiff
Supporting scripts and data for the analysis of real-time metabolism using <sup>13</sup>C NMR spectra of live _C. difficile_ cell cultures, [as published in Nature Chemical Biology (https://doi.org/10.1038/s41589-023-01275-9).](https://doi.org/10.1038/s41589-023-01275-9)

## Folder structure:
> [data](data) - datasets, including the updated `icdf843` metbolic model and the dFBA parameters  
> [etc](etc) - contains requirements.txt for a virtual environment to run the python code  
> [scripts](scripts) - all scripts used in analysis  
> > [MATLAB](scripts/MATLAB) - MATLAB scripts for visualizing the NMR spectra and dFBA analyses  
> > [process](scripts/process) - a python script and supporting csh scripts to process the raw NMR files and run the dFBA analyses  
 
## Installation
Clone this repository with `git clone https://github.com/Massachusetts-Host-Microbiome-Center/nmr-cdiff.git`

### Dependencies
1. The MATLAB functions require an installation of MATLAB. They were developed and tested with MATLAB R2019b.
2. The remaining analyses require python 3.8+ and a collection of packages. We have provided a `pip freeze` output [requirements.txt](etc/requirements.txt) that includes all of the python dependencies. The simplest way to load these dependencies is by creating a python virtual environment with `venv`, then installing all of the dependencies with `python -m pip install -r requirements.txt`.
3. [NMRpipe](https://www.ibbr.umd.edu/nmrpipe/install.html) is required to run the NMR processing scripts in process.py and any other functions that call them. Please follow the installation instructions for NMRpipe carefully.

## Tutorial
Follow [this tutorial](TUTORIAL.md) to learn how to process a test NMR dataset and run the dFBA analyses.

## License
This distribution is available under the [Apache License, Version 2.0](LICENSE).
