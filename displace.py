#!/usr/bin/env python

#
# create input files for different DFT codes
#

import sys, os
import numpy as np
from parserVASP import writePOSCAR
from parserPhonopy import parsePhonopy

def displace(modelist, stepsize, program, disps):
    # get phonon modes and unit cell
    eigvals, eigvecs, norms, qpoint, basis, nat, elements, positions = parsePhonopy(modelist)
    
    # write unit cells with displacements
    print("[displace]: Generating displacements...")
    for mode in modelist:
        eigvec = eigvecs[mode-1]
        norm = norms[mode-1]
        for disp in disps:
            file="mode"+str(mode)+"_"+str(disp)
            if os.path.isdir(file) == False:
                os.system("mkdir "+file)
            #
            if program == "VASP":
                writePOSCAR(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm)
            elif program == "QE":
                print("[displace]: Format not implemented, exiting")
                sys.exit(1)
            else:
                print("[displace]: Format not implemented, exiting")
                sys.exit(1)
            #
        #
        print("[displace]: Done.")
    #
#