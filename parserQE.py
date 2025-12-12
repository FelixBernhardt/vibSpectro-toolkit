#!/usr/bin/env python

#
# QuantumEspresso parsers
#

import sys, os
import numpy as np
from collections import Counter

def writeSCF(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, scffile):
    # read some information from scf-file
    # this has to include the &control, &system, &electrons, &ions, &cell, ATOMIC SPECIES, K_POINTS tags
    
    if os.path.isfile(scffile):
        try:
            ff = open(scffile, "r")
        except IOError:
            print("[writeSCF]: ERROR Couldn't open scf-inputfile, exiting...\n")
            sys.exit(1)
        #
        lines = [l.strip() for l in ff.readlines()] # Read the whole file removing tailoring spaces
        ff.close()
    #
    else:
        print("[writeSCF]: "+scffile+" not found, exiting...\n")
        sys.exit(1)
    # 

    control_start = lines.index(next(l for l in lines if len(l) > 0 if l.split()[0] == "&control" ))
    control_stop = lines.index(next(l for l in lines[control_start:] if len(l) > 0 if l.split()[0] == "/" ))
    control = lines[control_start:control_stop+1]

    system_start = lines.index(next(l for l in lines if len(l) > 0 if l.split()[0] == "&system" ))
    system_stop = lines[system_start:].index(next(l for l in lines[system_start:] if len(l) > 0 if l.split()[0] == "/" ))
    system = lines[system_start:system_start+system_stop+1]

    electrons_start = lines.index(next(l for l in lines if len(l) > 0 if l.split()[0] == "&electrons" ))
    electrons_stop = lines[electrons_start:].index(next(l for l in lines[electrons_start:] if len(l) > 0 if l.split()[0] == "/" ))
    electrons = lines[electrons_start:electrons_start+electrons_stop+1]

    ions_start = lines.index(next(l for l in lines if len(l) > 0 if l.split()[0] == "&ions" ))
    ions_stop = lines[ions_start:].index(next(l for l in lines[ions_start:] if len(l) > 0 if l.split()[0] == "/" ))
    ions = lines[ions_start:ions_start+ions_stop+1]

    cell_start = lines.index(next(l for l in lines if len(l) > 0 if l.split()[0] == "&cell" ))
    cell_stop = lines[cell_start:].index(next(l for l in lines[cell_start:] if len(l) > 0 if l.split()[0] == "/" ))
    cell = lines[cell_start:cell_start+cell_stop+1]

    as_start = lines.index(next(l for l in lines if len(l) > 0 if l.split()[0] == "ATOMIC_SPECIES" ))
    ATOMIC_SPECIES = lines[as_start:as_start+len(Counter(elements).keys())+1]

    k_start = lines.index(next(l for l in lines if len(l) > 0 if l.split()[0] == "K_POINTS" ))
    K_POINTS = lines[k_start:k_start+2]

    # write the given information
    #
    scf_fh = open(file+"/scf.in", "w")
    for i in range(len((control))):
        scf_fh.write( control[i]+"\n" )
    for i in range(len((system))):
        scf_fh.write( system[i]+"\n" )
    for i in range(len((electrons))):
        scf_fh.write( electrons[i]+"\n" )
    for i in range(len((ions))):
        scf_fh.write( ions[i]+"\n" )
    for i in range(len((cell))):
        scf_fh.write( cell[i]+"\n" )
    for i in range(len((K_POINTS))):
        scf_fh.write( K_POINTS[i]+"\n" )
    for i in range(len((ATOMIC_SPECIES))):
        scf_fh.write( ATOMIC_SPECIES[i]+"\n" )

    # write the displacements
    # basis
    scf_fh.write("CELL_PARAMETERS (angstrom)\n")
    for i in range(3):
        base = [ basis[i][l] for l in range(3)]
        scf_fh.write( "%15.10f %15.10f %15.10f\n" % (base[0], base[1], base[2]) )
    #
    
    scf_fh.write("ATOMIC_POSITIONS (cartesian)\n")
    for i in range(nat):
        pos_disp = [ positions[i][l] + eigvec[i][l]*stepsize*disp/norm for l in range(3)]
        scf_fh.write( "%s %15.10f %15.10f %15.10f\n" % (elements[i], pos_disp[0], pos_disp[1], pos_disp[2]) )
    #
    scf_fh.close()
#