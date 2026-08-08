#!/bin/bash

# example usage of the command line interface script to interface the python API
pwd=$(pwd)

# symmetry analysis
VibSpectro-toolkit --path=${pwd}+"/LiNbO3" --file="OUTCAR" --analysis

# caluate IR spectrum
VibSpectro-toolkit --path=${pwd}+"/LiNbO3" --file="OUTCAR" --infrared --reflectance --plotIR --plotReflectance

# Raman spectrum
# create displacements
VibSpectro-toolkit --path=${pwd}+"/LiNbO3" --file="OUTCAR" --displace --code_out="VASP"

# calculate the raman tensors, raman spectrum, and plot the results
VibSpectro-toolkit --path=${pwd}+"/LiNbO3" --file="OUTCAR" --tensors --spectrum --plotRaman --code_out="VASP"

# view additional flags
VibSpectro-toolkit --help
