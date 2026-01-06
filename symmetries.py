#!/usr/bin/env python

#
# interface to spglib to extract symmetry information
#

import sys, os
import numpy as np
from spglib import get_symmetry_dataset
import yaml
from parserPhonopy import parsePhonopy
from RamanLib import RamanTensorComponents, dielectricFunctionComponents, RamanSelectionRules, IRSelectionRules, formatString, periodTable, backDirs, rightDirs, IRDirs

def analyzeRamanTensors(pointgroup):
    RamanTensors = RamanTensorComponents(pointgroup)

    # print to console
    print("Raman Tensors of point group "+pointgroup)
    for i in range(int(len(RamanTensors)/2)):
        print(RamanTensors[2*i])
        for j in range(3):
            print(RamanTensors[2*i+1][j])
        #
    #
    print("")

    return RamanTensors
#

def analyzeDielectricTensor(pointgroup):
    dielectricTensor = dielectricFunctionComponents(pointgroup)

    # print to console
    print("Dielectric Tensor of point group "+pointgroup)
    for j in range(3):
        print(dielectricTensor[j])
    #
    print("")

    return dielectricTensor
#

def RamanSelection(pointgroup, RamanTensors):
    backscattering, backComponents, rightscattering, rightComponents = RamanSelectionRules(pointgroup, RamanTensors)

    # format the components
    for j in range(9):
        backComponents[j] = formatString(backComponents[j])
    #
    for j in range(8):
        rightComponents[j] = formatString(rightComponents[j])
    #               

    # print to console
    maxlen = np.max(np.concatenate(([len(x) for x in backscattering], [len(x) for x in rightscattering], [len("observable modes")])))
    print("Raman selection Rules for pointgroup "+pointgroup)
    placeholder1 = " " * int(np.ceil(np.abs(maxlen - len("observable modes"))/2))
    placeholder2 = " " * int(np.floor(np.abs(maxlen - len("observable modes"))/2))
    header = "        | "+placeholder1+"observable modes"+placeholder2+" | tensor components"
    print(header)
    for j in range(len(backDirs)):
        placeholder1 = " " * int(np.ceil(np.abs(maxlen - len(backscattering[j]))/2))
        placeholder2 = " " * int(np.floor(np.abs(maxlen - len(backscattering[j]))/2))
        print(" " + backDirs[j] + " | " + placeholder1 + backscattering[j] + placeholder2 + " | " + backComponents[j] )
    #
    print("-"*len(header))
    for j in range(len(rightDirs)):
        placeholder1 = " " * int(np.ceil(np.abs(maxlen - len(rightscattering[j]))/2))
        placeholder2 = " " * int(np.floor(np.abs(maxlen - len(rightscattering[j]))/2))
        print(" " + rightDirs[j] + " | " + placeholder1 + rightscattering[j] + placeholder2 + " | " + rightComponents[j] )
    #
    print("")
#

def IRSelection(pointgroup):
    scattering = IRSelectionRules(pointgroup)

    maxlen = np.max(np.concatenate(([len(x) for x in scattering], [len("observable modes")])))
    print("IR selection Rules for pointgroup "+pointgroup)
    placeholder1 = " " * int(np.ceil(np.abs(maxlen - len("observable modes"))/2))
    placeholder2 = " " * int(np.floor(np.abs(maxlen - len("observable modes"))/2))
    header = "        | "+placeholder1+"observable modes"+placeholder2
    print(header)
    for j in range(len(IRDirs)):
        placeholder1 = " " * int(np.ceil(np.abs(maxlen - len(scattering[j]))/2))
        placeholder2 = " " * int(np.floor(np.abs(maxlen - len(scattering[j]))/2))
        print(" " + IRDirs[j] + " | " + placeholder1 + scattering[j] + placeholder2 )
    #
    print("")
#

"""
def loadDM(file):
    with open(file, "r") as stream:
        dataDM = yaml.safe_load(stream)
    dynmat_data = dataDM["phonon"][0]["dynamical_matrix"]
    return dynmat_data
#
"""
    
def analysis(modelist):

    phonopy_fh = open("qpoints.yaml", "r")
    eigvals, eigvecs, norms, qpoint, basis, nat, elements, cPos, masses = parsePhonopy(modelist, None)
    phonopy_fh.close()

    coord = np.empty((nat, 3))
    for atom in range(nat):
        coord[atom, :] = np.dot(np.linalg.inv(basis.T), cPos[atom, :])
    #
    dataset = get_symmetry_dataset((basis, coord, [periodTable[element] for element in elements]), symprec=1.e-5)
    print("[analysis]: Space Group "+str(dataset["number"]))
    print("[analysis]: Point Group "+dataset["pointgroup"]) 
    print(dataset["wyckoffs"])

    # get the corresponding Raman tensors and selection rules
    pointgroup = dataset["pointgroup"]
    RamanTensors = analyzeRamanTensors(pointgroup)
    RamanSelection(pointgroup, RamanTensors)
    dielectricTensor = analyzeDielectricTensor(pointgroup)
    IRSelection(pointgroup)
    #RamanTensors = analyzeRamanTensors("m-3m")
    #RamanSelection("m-3m", RamanTensors)
    #dielectricTensor = analyzeDielectricTensor("m-3m")
    #IRSelection("m-3m")
    decomposition()

    """
    # set up a phonopy structure and get irreps
    import phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.phonon.irreps import IrReps
    #from phonopy.file_IO import read_force_constants

    cell = PhonopyAtoms( symbols=elements, cell=basis, scaled_positions=coord )
    phonopy_instance = phonopy.load(unitcell=cell, supercell_matrix=np.eye(3), primitive_matrix="auto", force_constants_filename="FORCE_CONSTANTS")
    irreps = IrReps(phonopy_instance, qpoint)
    
    # continue...?
    """


    sys.exit(1)