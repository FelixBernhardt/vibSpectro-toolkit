#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import sys
# smoothen dielectric function in "read_optics"
#from scipy.signal import savgol_filter
from RamanLib import *

def calcTensors(modeList, programIN, programOUT):
    # get phonon modes and unit cell
    if programIN == "VASP":
        from parserVASP import parsePOSCAR, parseOUTCAR

        poscar_fh = open("POSCAR.phon", "r")
        nat, vol, b, pos, poscar_header, num_atoms, atom_types = parsePOSCAR(poscar_fh)
        poscar_fh.close()
        outcar_fh = open("OUTCAR.phon", "r")
        eigvals, eigvecs, norms = parseOUTCAR(outcar_fh, nat)
        outcar_fh.close()
    #
    elif programIN == "QE":
        print("[calcTensors]: Format not implemented, exiting")
    #
    elif programIN == "phonopy":
        from parserPhonopy import parsePhonopy
        phonopy_fh = open("qpoints.yaml", "r")
        eigvals, eigvecs, norms = parsePhonopy(phonopy_fh)
        phonopy_fh.close()

    else:
        print("[calcTensors]: Format not implemented, exiting")
    #


    print("[calcTensors]: Calculating Raman tensors...")
    
    if programOUT == "VASP":
        from parserVASP import parseOptics
        #file_check(modeList, "vasprun", LO_dir)
        breakout = 0
        iteration = 0
        total = len(modeList)
        for mode in modeList:
            printProgressBar(iteration, total-1)
            eigval = eigvals[mode-1]
            eigvec = eigvecs[mode-1]
            norm = norms[mode-1]
            breakout = parseOptics("vasprun"+str(mode)+"_1", breakout)
            w1, Im1, Re1 = read_optics("optics.dat")
            breakout = parseOptics("vasprun"+str(mode)+"_-1", breakout)
            w2, Im2, Re2 = read_optics("optics.dat")
            if breakout == 0:
                #print("[calcTensors]: Calculating mode "+str(mode))
                w, Im1, Re1, Im2, Re2 = align_omega(w1, w2, Im1, Re1, Im2, Re2)
                calc_raman(mode, eigval, w, Im1, Re1, Im2, Re2)
            #
            iteration += 1
        #
        os.system("rm grep_optics.sh")
        os.system("rm optics.dat")
        if breakout == 1:
            sys.exit(1)
        #
        print("[__main__]: Done.")
        sys.exit(1)
    #