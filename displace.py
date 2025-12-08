#!/usr/bin/env python

#
# create input files for different DFT codes
#

import sys
import numpy as np
#from parserVASP import writePOSCAR
from parserPhonopy import parsePhonopy

def displace(modeList, stepsize, program):
    disps = [-1, 1]
    # get phonon modes and unit cell
    eigvals, eigvecs, norms, qpoint, basis, nat, elements, pos = parsePhonopy()

    # check
    if 3*nat < np.max(modeList):
        print("[displace]: invalid mode specified, check your input files for consistency, exiting...")
        sys.exit(1)
    #
    
    # write unit cells with displacements
    print("[displace]: Generating displacements...")
    for mode in modeList:
        eigval = eigvals[mode-1]
        eigvec = eigvecs[mode-1]
        norm = norms[mode-1]
        file="mode"+str(mode)
        if program == "VASP":
            writePOSCAR(mode, stepsize, norm, eigvec, basis, nat, elements, pos, file)
        elif program == "QE":
            print("[displace]: Format not implemented, exiting")
            sys.exit(1)
        else:
            print("[displace]: Format not implemented, exiting")
            sys.exit(1)
        #
        print("[displace]: Done.")
    #
#