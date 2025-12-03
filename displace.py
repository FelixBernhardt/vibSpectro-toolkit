#!/usr/bin/env python

#
# create input files for different DFT codes
#

import re
import sys
import os.path
from math import sqrt
import numpy as np
from RamanLib import flatten, MAT_m_VEC, T

def displace(modeList, disps, stepSize, programIN, programOUT):
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
    if programOUT == "VASP":
        print("[displace]: Generating POSCARs...")
        for mode in modeList:
            eigval = eigvals[mode-1]
            eigvec = eigvecs[mode-1]
            norm = norms[mode-1]

            for j in disps:
                poscar_fh = open("POSCAR_"+str(mode)+"_"+str(j), 'w')
                poscar_fh.write("%s %4.1e \n" % (str(mode)+"  "+str(j), stepSize))
                poscar_fh.write(poscar_header)
                poscar_fh.write("Cartesian\n")
                #
                for k in range(nat):
                    pos_disp = [ pos[k][l] + eigvec[k][l]*stepSize*j/norm for l in range(3)]
                    poscar_fh.write( '%15.10f %15.10f %15.10f\n' % (pos_disp[0], pos_disp[1], pos_disp[2]) )
                #
            #
            poscar_fh.close()
        #
        print("[displace]: Done.")
    #
    elif programOUT == "QE":
        print("[displace]: Generating QE input structures...")
        for mode in modeList:
            eigval = eigvals[mode-1]
            eigvec = eigvecs[mode-1]
            norm = norms[mode-1]

            for j in disps:
                pw_fh = open("scf_"+str(mode)+"_"+str(j), 'w')
                pw_fh.write("%s %4.1e \n" % (str(mode)+"  "+str(j), stepSize))
                pw_fh.write(QE_header)
                pw_fh.write("Cartesian\n")
                #
                for k in range(nat):
                    pos_disp = [ pos[k][l] + eigvec[k][l]*stepSize*j/norm for l in range(3)]
                    pw_fh.write( '%15.10f %15.10f %15.10f\n' % (pos_disp[0], pos_disp[1], pos_disp[2]) )
                #
            #
            pw_fh.close()
        #
        print("[displace]: Done.")



        print("[displace]: Format not implemented, exiting")
    #
    else:
        print("[displace]: Format not implemented, exiting")
    #
    sys.exit(1)
#