#!/usr/bin/env python

#
# argument parser for RamanPy
#

import sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--version", action="store_true",\
                    help="prints the version number.")
parser.add_argument("-a", "--analysis", action="store_true",\
                    help="analyzes the symmetries of the structure and prints information.")
parser.add_argument("-m", "--modelist", type=str, default=None,\
                    help="The phonon modes to be considered for the calculations. \n\
                    The labelling is in ascending order according to the mode's frequencies. \
                    Format of modelist:\
                    - single modes: integers separated by blank \" \"\
                    - range of modes: lower and upper limit separated by hyphen \"-\"\
                     example: '3-6 8 11-15' will consider modes 3, 4, 5, 6, 8, 11, 12, 13, 14, 15")
parser.add_argument("-d", "--displace", action="store_true",\
                    help="add the ionic displacements according to the phonon modes provided by the --modelist option.")
parser.add_argument("-t", "--tensors", action="store_true",\
                    help="Calculate the Raman tensors of the phonon modes provided by the --modelist option")
parser.add_argument("-s", "--spectrum", action="store_true",\
                    help="Calculate the Raman spectrum for all polarization directions of all phonon modes provided by the --modelist option.")
parser.add_argument("-p", "--plot", action="store_true",\
                    help="plot the Raman spectrum of the configuration specified using the --porto option. Only the phonon modes considered for the calculation of the spectrum are considered.")
parser.add_argument("-IR", "--infrared", action="store_true",\
                    help="calculate the IR spectrum of the phonon modes provided by the --modelist option.")
parser.add_argument("-LO", "--LOcorr", action="store_true",\
                    help="incorporate an LO correction term when calculating the Raman spectrum.")
parser.add_argument("-w", "--laser", type=float, default=2.0,\
                    help="The excitation energy used for calculating the spectrum (eV)")
parser.add_argument("-tp", "--temperature", type=float, default=300.0,\
                    help="The temperature used for calculating the spectrum (K)")
parser.add_argument("-sm", "--smearing", type=float, default=5.0,\
                    help="The smearing used for calculating the spectrum (cm^-1)")
parser.add_argument("-sz", "--stepsize", type=float, default=0.001,\
                    help="The prefactor for the displacements used for generating the input files (unit?)")
parser.add_argument("-pt", "--porto", type=str, default="xx",\
                    help="The polarization used when plotting the spectrum using porto's notation\
                         .(xx). , where \"xx\" can be set to all combinations of cartesian directions.\
                         A spatially averaged spectrum can be plotted by setting to \"avg\".")
parser.add_argument("-q", "--qdir", type=str, default="0 0 1",\
                    help="The propagation direction used when plotting the spectrum using porto's notation\
                         q(..)q , where \"qq\" can be set to backscattering or right angle scattering (without the sign).\
                         Only used in combination with the LO flag.")
parser.add_argument("-P", "--program", type=str, default="VASP",\
                    help="The software package to write the created cells into and read the dielectric function from.\
                          currently supported: VASP (default), QE")
parser.add_argument("-scf", "--QEinputfile", type=str, default="scf.in",\
                    help="The pw.x input file to be duplicated for the Raman calculations")
parser.add_argument("-shg", "--nonlincorr", type=str, default=None,\
                    help="Reads in the SHG tensor from file")
parser.add_argument("-nosym", "--no_symmetry", action="store_true",\
                    help="ignores symmetries and explicitly calculates all given modes")
parser.add_argument("-allVASP", "--readFromVASP", action="store_true",\
                    help="get the cell and phonon information directly from POSCAR and OUTCAR")

args = parser.parse_args()

# weird but works
args.qdir = tuple(map(int, args.qdir.split()))
"""
    if opt == "h":
        print("A single script to calculate Raman spectra with external DFT codes")
        print("needs following files (VASP used as an example):")
        print("- POSCAR.phon     : the equilibrium structure, for which all phonon modes and frequencies are already calculated")
        print("- OUTCAR.phon     : output containing the phonon eigenmodes and frequencies")

        print("Calculation steps:")
        print("1  : run 'Raman.py -d -m=<modeList>")
        print("2  : run VASP for all the generated POSCARs in their respective folders")
        print("4  : run Raman.py -t -m=<modeList>")
        print("     this will calculate the frequency dependent Raman tensors")
        print("5  : run Raman.py -s -m=<modeList> -w=<float> -tp=<float> -sm=<float>")
        print("     set the parameters for the Raman spectrum calculation")
        print("     excitation frequency wavelength (eV), temperature (K), smearing width (cm-1)")
        print("optional steps:")
        print("6    run 'Raman.py -p -pt=<str>")
        print("     set the polarization direction (Porto notation, i.e. xx)") 
        print("     check the selection rules to only include TO modes, e.g. here")
        print("     https://www.cryst.ehu.es/cryst/polarizationselrules.html")
        print("     the Raman intensities for LO-modes are not implemented yet!)
       
    #
"""