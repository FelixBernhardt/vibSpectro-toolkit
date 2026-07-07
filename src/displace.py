#!/usr/bin/env python

#
# create input files for different DFT codes
#

import os
import numpy as np

def calcDisplace(path, modelist, stepsize, parser, eigvecs, norms, basis, nat, elements, positions, scffile):
    
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
            parser.write_file(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, scffile)
            parser.link_file(file)
        #
    #
    print("[calcDisplace]: Done.")
#