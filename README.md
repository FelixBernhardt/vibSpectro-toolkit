
# Vibrational Spectroscopy toolkit
Some utility scripts to calculate and analyze IR- and Raman spectra.

## Features
This repository provides a python framework, as well as a simple command line script to:
- calculate IR spectra and reflectivity
- calculate Raman spectra (resonant and non-resonant, stokes and anti-stokes, adjustable linewidth/temperature/photon frequency)
- analyze symmetry configurations and consider LO-TO splitting (only in conjunction with phonopy)
- output the simulated data and spectra
The script requires at the Gamma point calculated phononic eigenmodes and frequencies. Supported interfaces include phonopy as input (and thus also all calculators supported by phonopy), and VASP and QuantumEspresso for output. Additional calculators can easily be implemented in the provided framework.
<br>
Check the examples folder for tutorials on how to perform the calculations.

## Installation

Install all the libraries listed in the requirements.txt file. Place the files of this script in whatever directory you want.

You can access the scripts by adding the following line in ~/.bashrc:
```bash
export PATH="${PATH}:path_to_your_install_directory/vibSpectro-toolkit"
export PYTHONPATH="${PYTHONPATH$:}:path_to_your_install_directory/vibSpectro-toolkit"
```

## Cite
There currently exists no standalone paper for this project. If you use this software, please consider citing one of the following when publishing your results:
- this github repository
- https://doi.org/10.1002/pssa.202300968
- https://doi.org/10.1021/acs.jpcc.4c05225 
