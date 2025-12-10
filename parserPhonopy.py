#!/usr/bin/env python

#
# parser for phonopy
#

import sys
import numpy as np
import yaml

"""
unit conventions exported from phonopy documentation at https://phonopy.github.io/phonopy/interfaces.html

          | Distance   Atomic mass   Force         Force constants
-----------------------------------------------------------------
VASP      | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
WIEN2k    | au (bohr)  AMU           mRy/au        mRy/au^2
QE        | au (bohr)  AMU           Ry/au         Ry/au^2
ABINIT    | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
SIESTA    | au (bohr)  AMU           eV/Angstrom   eV/Angstrom.au
Elk       | au (bohr)  AMU           hartree/au    hartree/au^2
CRYSTAL   | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
TURBOMOLE | au (bohr)  AMU           hartree/au    hartree/au^2
CP2K      | Angstrom   AMU           hartree/au    hartree/Angstrom.au
FHI-AIMS  | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
Fleur     | au (bohr)  AMU           hartree/au    hartree/au^2
CASTEP    | Angstrom   AMU           eV/angstrom   eV/angstrom^2
ABACUS    | au (bohr)  AMU           eV/angstrom   eV/angstrom.au
LAMMPS    | Angstrom   AMU           eV/Angstrom   eV/Angstrom^2
QLM       | au (bohr)  AMU           Ry/au         Ry/au^2
"""

# conversions from WolframAlpha https://www.wolframalpha.com

eV2J = 1.602e-19
angstrom2m = 1.e-10
amu2kg = 1.66053907e-27

eV2rcm = 8066
au2angstrom = 0.529177211
Ry2eV = 13.605693123
mRy2eV = Ry2eV/1000
hartree2eV = 27.211386246

V2THz = 15.633302
THz2cm = 33.36


def parsePhonopy(modelist):
    with open("qpoints.yaml", "r") as stream:
        dataDM = yaml.safe_load(stream)
    with open("phonopy.yaml", "r") as stream:
        dataC = yaml.safe_load(stream)
    #

    # parse the unit cell information 

    # get the units
    #mass = dataC["physical_unit"]["atomic_mass"]
    length = dataC["physical_unit"]["length"]
    force_constants = dataC["physical_unit"]["force_constants"]

    # get the symmetries just to check
    print("[parsePhonopy]: space group "+dataC["space_group"]["type"])

    # cell data
    basis = np.array(dataC["primitive_cell"]["lattice"])
    nat = len(dataC["primitive_cell"]["points"])
    pos = np.empty((nat, 3))
    masses = np.empty(nat)
    elements = []
    for j in range(nat):
        pos[j] = dataC["primitive_cell"]["points"][j]["coordinates"]
        masses[j] = dataC["primitive_cell"]["points"][j]["mass"]
        elements.append( dataC["primitive_cell"]["points"][j]["symbol"] )
    #

    # checks
    if 3*nat < np.max(modelist):
        print("[parsePhonopy]]: invalid mode specified, check your input files for consistency, exiting...")
        sys.exit(1)
    #
    nat2 = dataDM["natom"]
    if nat != nat2:
        print("[parsePhonopy]: phonopy.yaml and qpoints.yaml files don't match, exiting...")
        sys.exit(1)
    #

    # parse the dynamical matrix
    # if multiple qpoints are present, only use the first one
    qpoint = dataDM["phonon"][0]["q-position"]
    print("[parsePhonopy]: q-point "+str(qpoint))
    dynmat = []
    dynmat_data = dataDM["phonon"][0]["dynamical_matrix"]
    for row in dynmat_data:
        vals = np.reshape(row, (-1, 2))
        dynmat.append(vals[:, 0] + vals[:, 1] * 1j)
    dynmat = np.array(dynmat)
    eigvals, eigvecs_tmp, = np.linalg.eigh(dynmat)
    frequencies = (np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real)) /np.sqrt(amu2kg) / angstrom2m / (2*np.pi) /1e12 * THz2cm
    eigvecs =  np.zeros((3*nat,nat,3))
    for j in range(3*nat):
        eigvecs[j] = np.reshape(eigvecs_tmp[:,nat-j-1].real, (nat,3))
    #
    eigvecsNorm = np.empty_like(eigvecs)
    norms = np.empty(3*nat)
    for j in range(len(eigvecs)):
        for atom in range(nat):
            eigvecsNorm[j][atom] = eigvecs[j][atom]/np.sqrt(masses[atom])
            #
        #
        norms[j] = np.linalg.norm(eigvecsNorm[j])
    #

    # convert all to VASP default units, phonon frequncies to cm^-1
    # length in angstrom
    if length == "au":
        basis = au2angstrom*basis
    elif length == "angstrom":
        pass
    else:
        print("[parsePhonopy]: The unit "+length+" is currently not supported for lengths, exiting...")
        sys.exit(1)
    #
    # force constants in eV/angstrom^2
    # frequencies in cm^-1
    if force_constants == "Ry/au^2":
        frequencies = frequencies*np.sqrt(Ry2eV*eV2J)*au2angstrom
    elif force_constants == "mRy/au^2":
        frequencies = frequencies*np.sqrt(mRy2eV*eV2J)*au2angstrom
    elif force_constants == "eV/Angstrom.au":
        frequencies = frequencies*np.sqrt(eV2J)*np.sqrt(angstrom2m/au2angstrom)
    elif force_constants == "hartree/au^2":
        frequencies = frequencies*np.sqrt(hartree2eV*eV2J)*au2angstrom
    elif force_constants == "hartree/Angstrom.au":
        frequencies = frequencies*np.sqrt(hartree2eV*eV2J)*np.sqrt(angstrom2m/au2angstrom)
    elif force_constants == "eV/angstrom^2":
        frequencies = frequencies*np.sqrt(eV2J)
    else:
        print("[parsePhonopy]: The unit "+force_constants+" is currently not supported for force constants, exiting...")
        sys.exit(1)
    #

    # all positions to cartesian coordinates
    cPos = np.empty_like(pos)
    for j in range(nat):
        cPos[j] = pos[j][0]*basis[0] + pos[j][1]*basis[1] + pos[j][2]*basis[2]
    #

    #print(eigvals)
    #print(eigvecs_norm[0])
    #print(norms)

    print(eigvecsNorm)

    return list(reversed(frequencies)), eigvecsNorm, norms, qpoint, basis, nat, elements, cPos
#