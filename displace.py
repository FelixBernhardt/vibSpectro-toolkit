#!/usr/bin/env python

#
# create input files for different DFT codes
#

import sys, os
from parserVASP import writePOSCAR, linkVASP
from parserQE import writeSCF
from parserPhonopy import parsePhonopy

def displace(modelist, stepsize, program, disps, scffile):
    # get phonon modes and unit cell
    eigvals, eigvecs, norms, qpoint, basis, nat, elements, positions, masses = parsePhonopy(modelist)
    
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
                print(eigvals[mode-1])
                print(eigvec)
                print(norm)
                writePOSCAR(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm)
                linkVASP(file)
            elif program == "QE":
                writeSCF(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, scffile)
            else:
                print("[displace]: Format not implemented, exiting")
                sys.exit(1)
            #
        #
    #
    print("[displace]: Done.")
#