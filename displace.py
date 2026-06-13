#!/usr/bin/env python

#
# create input files for different DFT codes
#

import sys, os
import numpy as np
from parserVASP import writePOSCAR, linkVASP
from parserQE import writeSCF, linkQE

def calcDisplace(path, modelist, stepsize, program, eigvecs, norms, basis, nat, elements, positions, scffile):
    if program != "VASP" and program != "QE":
        print("[displace]: code not supported, exiting...")
        sys.exit(1)
    #
    # write unit cells with displacements
    disps = [-1, 1]
    print("[displace]: Generating displacements...")
    if os.path.isdir(path+"displacements") == False:
        os.system("mkdir "+path+"displacements")
    #
    for mode in modelist:
        
        eigvec = np.real(eigvecs[mode])
        
        norm = norms[mode]
        for disp in disps:
            file=path+"displacements/mode"+str(mode)+"_"+str(disp)
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