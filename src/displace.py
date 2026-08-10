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
    # df/dx = 1/2h [ f(x+h)-f(x-h) ]
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

def calcDisplaceSecondOrder(path, combinedmodelist, stepsize, parser, eigvecs, basis, nat, elements, positions, scffile):
    
    # write unit cells with displacements
    disps1 = [-1, 1]
    disps2 = [-1, 1]
    print("[calcDisplaceSecondOrder]: Generating displacements...")
    if os.path.isdir(path+"displacements") == False:
        os.system("mkdir "+path+"displacements")
    #
    for mode in combinedmodelist:
        
        eigvec1 = np.real(eigvecs[mode[0]])
        eigvec2 = np.real(eigvecs[mode[1]])

        # df/dxdy = 1/4hk [ f(x+k,y+h)-f(x-k,y+h)-f(x+k,y-h)+f(x-k,y-h) ]
        for disp1 in disps1:
            for disp2 in disps2:
                file=path+"displacements/mode"+str(mode[0])+"_"+str(mode[1])+"_"+str(disp1)+"_"+str(disp2)
                if os.path.isdir(file) == False:
                    os.system("mkdir "+file)
                #
                displacement = disp1*eigvec1+disp2*eigvec2
                norm = np.linalg.norm(displacement)
                parser.write_file(nat, basis, positions, elements, file, mode, 1, stepsize, displacement, norm, scffile)
                parser.link_file(file)
        #
    #

    print("[calcDisplaceSecondOrder]: Done.")
#