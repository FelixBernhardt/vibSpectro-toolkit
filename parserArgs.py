#!/usr/bin/env python

#
# argument parser for RamanPy
#

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=str, default="./",\
                    help="The path where the input information can be found.")
parser.add_argument("-f", "--file", type=str, default="OUTCAR",\
                    help="The file where the input information can be found.")
parser.add_argument("-m", "--modelist", type=str, default=None,\
                    help="The phonon modes to be considered for the calculations. \n\
                    The labelling is in ascending order according to the mode's frequencies. \
                    Format of modelist:\
                    - single modes: integers separated by blank \" \"\
                    - range of modes: lower and upper limit separated by hyphen \"-\"\
                     example: '3-6 8 11-15' will consider modes 3, 4, 5, 6, 8, 11, 12, 13, 14, 15")
parser.add_argument("-LO", "--LOcorr", type=bool, default=False,\
                    help="Incorporate an LO correction term when calculating the Raman spectrum.")
parser.add_argument("-w", "--laser", type=float, default=2.0,\
                    help="The excitation energy used for calculating the spectrum (eV).")
parser.add_argument("-tp", "--temperature", type=float, default=300.0,\
                    help="The temperature used for calculating the spectrum (K).")
parser.add_argument("-sm", "--smearing", type=float, default=5.0,\
                    help="The smearing used for calculating the spectrum (cm^-1).")
parser.add_argument("-sz", "--stepsize", type=float, default=0.001,\
                    help="The prefactor for the displacements used for generating the input files (unit?)")
parser.add_argument("-pt", "--porto", type=str, default="xx",\
                    help="The polarization used when plotting the spectrum using porto's notation\
                         .(xx). , where \"xx\" can be set to all combinations of cartesian directions.\
                         A spatially averaged spectrum can be plotted by setting to \"avg\".")
parser.add_argument("-sto", "--stokes", type=str, default="stokes",\
                    help="Calculate Stokes (default) or anti-Stokes Raman Intensity.")
parser.add_argument("-q", "--qdir", type=str, default="1 0 0",\
                    help="The propagation direction used when plotting the spectrum using porto's notation\
                         q(..)q , where \"qq\" can be set to backscattering or right angle scattering (without the sign).\
                         Only used in combination with the LO flag.")
parser.add_argument("-P", "--code_out", type=str, default="VASP",\
                    help="The software package to write the created cells into and read the dielectric function from.\
                          currently supported: VASP (default), QE")
parser.add_argument("-scf", "--QEinputfile", type=str, default="scf.in",\
                    help="The pw.x input file to be duplicated for the Raman calculations.")
parser.add_argument("-nosym", "--no_symmetry", type=bool, default=False,\
                    help="Ignores symmetries and explicitly calculates all given modes.")
parser.add_argument("-mol", "--molecule", type=bool, default=False,\
                    help="Define if structure is a molecule.")
parser.add_argument("-lua", "--lualatex", type=bool, default=False,\
                    help="Use LuaLatex for plotting.")

parser.add_argument("-v", "--version", action="store_true",\
                    help="Prints the version number.")
parser.add_argument("-a", "--analysis", action="store_true",\
                    help="Analyzes the symmetries of the structure and prints information.")
parser.add_argument("-d", "--displace", action="store_true",\
                    help="Add the ionic displacements according to the phonon modes provided by the --modelist option.")
parser.add_argument("-t", "--tensors", action="store_true",\
                    help="Calculate the Raman tensors of the phonon modes provided by the --modelist option")
parser.add_argument("-s", "--spectrum", action="store_true",\
                    help="Calculate the Raman spectrum for all polarization directions of all phonon modes provided by the --modelist option.")
parser.add_argument("-pR", "--plotRaman", action="store_true",\
                    help="plot the Raman spectrum of the configuration specified using the --porto option. Only the phonon modes considered for the calculation of the spectrum are considered.")
parser.add_argument("-IR", "--infrared", action="store_true",\
                    help="Calculate the IR spectrum of the phonon modes provided by the --modelist option.")
parser.add_argument("-pI", "--plotIR", action="store_true",\
                    help="Plot the IR spectrum of the phonon modes provided by the --modelist option.")

args = parser.parse_args()

# weird but works
args.qdir = tuple(map(int, args.qdir.split()))