#!/usr/bin/env python

#
# create input files for different DFT codes
#

import sys
import numpy as np
from parserVASP import parsePOSCAR, parseOUTCAR, writePOSCAR

def displace(modeList, disps, stepSize, programIN, programOUT):
    # get phonon modes and unit cell
    if programIN == "VASP":
        outcar_fh = open("OUTCAR", "r")
        eigvals, eigvecs, norms = parseOUTCAR(outcar_fh)
        outcar_fh.close()
    #
    elif programIN == "QE":
        pw_fh = open("scf.out", "r")

        pw_fh.close()
        print("[displace]: Format not implemented, exiting")
    #
    elif programIN == "phonopy":
        from parserPhonopy import parsePhonopy
        phonopy_fh = open("qpoints.yaml", "r")
        eigvals, eigvecs, norms = parsePhonopy(phonopy_fh)
        phonopy_fh.close()

    else:
        print("[displace]: Format not implemented, exiting")
    #
    
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
        if programOUT == "VASP":
            writePOSCAR(mode, stepSize, norm, eigvec, pos, file)
        elif programOUT == "QE":
            print("[displace]: Format not implemented, exiting")
            sys.exit(1)
        else:
            print("[displace]: Format not implemented, exiting")
            sys.exit(1)
        #
        print("[displace]: Done.")
    #
#