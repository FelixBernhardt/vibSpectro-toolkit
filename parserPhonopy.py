#!/usr/bin/env python

#
# parser for phonopy
#

import sys
import numpy as np
import yaml

def parsePhonopy():
    with open("qpoints.yaml", "r") as stream:
        dataDM = yaml.safe_load(stream)
    with open("phonopy.yaml", "r") as stream:
        dataC = yaml.safe_load(stream)
    #

    # parse the unit cell information

    # get the units
    mass = dataC["physical_unit"]["atomic_mass"]
    length = dataC["physical_unit"]["length"]
    force_constants = dataC["physical_unit"]["force_constants"]

    # get the symmetries just to check
    print("[parsePhonopy]: found space group "+dataC["space_group"]["type"])

    # cell data
    basis = dataC["primitive_cell"]["lattice"]
    nat = len(dataC["primitive_cell"]["points"])
    pos = np.empty((nat, 3))
    masses = np.empty(nat)
    elements = []
    for j in range(nat):
        pos[j] = dataC["primitive_cell"]["points"][j]["coordinates"]
        masses[j] = dataC["primitive_cell"]["points"][j]["mass"]
        elements.append( dataC["primitive_cell"]["points"][j]["symbol"] )
    #

    # parse the dynamical matrix
    nat2 = dataDM["natom"]
    if nat != nat2:
        print("[parsePhonopy]: phonopy.yaml and qpoints.yaml files don't match, exiting...")
        sys.exit(1)
    #

    qpoint = dataDM["phonon"][0]["q-position"]
    dynmat = []
    dynmat_data = dataDM["phonon"][0]["dynamical_matrix"]
    for row in dynmat_data:
        vals = np.reshape(row, (-1, 2))
        dynmat.append(vals[:, 0] + vals[:, 1] * 1j)
    dynmat = np.array(dynmat)
    eigvals, eigvecs_tmp, = np.linalg.eigh(dynmat)
    frequencies = (np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real))
    eigvecs =  np.zeros((len(eigvals),int(len(eigvals)/3),3))
    for j in range(len(eigvals)):
        eigvecs[j] = np.reshape(eigvecs_tmp[:,len(eigvals)-j-1], newshape=(int(len(eigvals)/3),3))
    #
    eigvecs_new = np.empty_like(eigvecs)
    norms = np.empty(3*nat)
    for j in range(len(eigvecs)):
        for atom in range(nat):
            eigvecs_new[j][atom] = eigvecs[j][atom]/np.sqrt(masses[atom])
            #
        #
        norms[j] = np.linalg.norm(eigvecs_new[j])
    #    
    return list(reversed(frequencies)), eigvecs_new, norms, qpoint, basis, nat, elements, pos
#