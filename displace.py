#!/usr/bin/env python

#
# create input files for different DFT codes
#

import sys, os
import numpy as np
from parserVASP import writePOSCAR, linkVASP
from parserQE import writeSCF, linkQE
from parserPhonopy import parsePhonopy
from RamanLib import removeModes

def displace(modelist_orig, stepsize, program, disps, scffile):
    # get phonon modes and unit cell
    eigvals, eigvecs, norms, qpoint, basis, nat, elements, positions, masses = parsePhonopy(modelist_orig, None)
    modelist = removeModes(eigvecs, eigvals, masses, modelist_orig)


    # write unit cells with displacements
    print("[displace]: Generating displacements...")
    if os.path.isdir("displacements") == False:
        os.system("mkdir displacements")
    #
    for mode in modelist:
        
        eigvec = np.real(eigvecs[mode-1])
        
        norm = norms[mode-1]
        for disp in disps:
            file="displacements/mode"+str(mode)+"_"+str(disp)
            if os.path.isdir(file) == False:
                os.system("mkdir "+file)
            #
            if program == "VASP":
                writePOSCAR(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm)
                linkVASP(file)
            elif program == "QE":
                writeSCF(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, scffile)
                linkQE(file)
            else:
                print("[displace]: Format not implemented, exiting")
                sys.exit(1)
            #
        #
    #
    print("[displace]: Done.")
#