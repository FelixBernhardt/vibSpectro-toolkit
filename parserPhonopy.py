#!/usr/bin/env python

#
# parser for phonopy
#

import re
import sys
import os.path
from math import sqrt
import numpy as np
import yaml
from phonopy.units import THzToCm
from RamanLib import masses

def parsePhonopy(file):
    conversion_factor_to_THz = 15.633302
    with open(file, 'r') as stream:
        data = yaml.safe_load(stream)
    #
    dynmat = []
    dynmat_data = data['phonon'][0]['dynamical_matrix']
    for row in dynmat_data:
        vals = np.reshape(row, (-1, 2))
        dynmat.append(vals[:, 0] + vals[:, 1] * 1j)
    dynmat = np.array(dynmat)
    eigvals, eigvecs_tmp, = np.linalg.eigh(dynmat)
    frequencies = (np.sqrt(np.abs(eigvals.real)) * np.sign(eigvals.real))*THzToCm*conversion_factor_to_THz
    eigvecs =  np.zeros((len(eigvals),int(len(eigvals)/3),3))
    for j in range(len(eigvals)):
        eigvecs[j] = np.reshape(eigvecs_tmp[:,len(eigvals)-j-1], newshape=(int(len(eigvals)/3),3))
    #
    eigvecs_new = np.empty_like(eigvecs)
    norms = np.empty(3*numAtoms)
    for j in range(len(eigvecs)):
        type = 0
        for atom in range(int(len(eigvecs)/3)):
            if atom >= sum(numAtoms[:type+1]):
                type += 1
            #
            eigvecs_new[j][atom] = eigvecs[j][atom]/np.sqrt(masses[atom_types[type]])
            #
        #
        norms[j] = np.linalg.norm(eigvecs_new[j])
    #    
    return list(reversed(frequencies)), eigvecs_new, norms
#

"""
# LO mode stuff not working
def phonopy_assign(eigvecs1, eigvals1, eigvecs2, eigvals2, nat):
    # assign phonopy LO-TO splitting
    keys = []
    values = []
    assigned = []
    certain = []
    for i in range(3*nat):
        prod = []
        for j in range(3*nat):
            if j not in assigned:
            #if (eigvals2[j] - eigvals1[i]) > -1000 :
                prod.append(np.abs(np.dot(flatten(eigvecs1[i]), flatten(eigvecs2[j]))))
            else:
                prod.append(0.0)
            #else:
            #    prod.append(0.0)
            #
        index = max(range(len(prod)), key=prod.__getitem__)
        #if prod[index] < 0.9:
        #    print(prod[index])
        #    print(prod)
        #    print(str(eigvals1[i]) + " -> " + str(eigvals2[index]))
        certain.append(prod[index])
        assigned.append(index)
        keys.append(i)
        values.append(index)
        sum = 0
        for j in range(len(prod)):
            sum += np.abs(prod[j])**2
        #print(sum, np.abs(np.dot(flatten(eigvecs1[i]), flatten(eigvecs1[i])))**2 )
    #
    print(certain)
    print(np.mean(certain))
    #print(np.min(certain))
    # only one assignment is to be allowed with a certainty < 0.5
    counter = 0
    for j in certain:
        if j < 0.1:
            counter += 1
        #
    #
    #if counter > 2:
    #    print("ERROR: THERE IS A PROBLEM ASSIGNING THE MODES")
    #    sys.exit(1)
    #
    mode_dict = dict(zip(keys, values))
    print(mode_dict)
    return mode_dict
#
"""