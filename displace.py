#!/usr/bin/env python

#
# create input files for different DFT codes
#

import os
import numpy as np
from parserVASP import writePOSCAR, linkVASP
from parserQE import writeSCF, linkQE

def calcDisplace(path, modelist, stepsize, program, eigvecs, norms, basis, nat, elements, positions, scffile):
    if program != "VASP" and program != "QE":
        print("[calcDisplace]: ERROR, code not supported")
    #
    # write unit cells with displacements
    disps = [-1, 1]
    print("[calcDisplace]: Generating displacements...")
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
                print("[calcDisplace]: ERROR, format not implemented")
            #
        #
    #
    print("[calcDisplace]: Done.")
#