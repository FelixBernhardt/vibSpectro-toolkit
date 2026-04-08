#!/usr/bin/env python

#
# contains the phonon class storing all needed information
#
import numpy as np
from numpy.typing import NDArray
from RamanLib import periodTable, periodTableMasses, getAcoustics, getDegenerates, getSilent
from symmetries import *
from displace import displace

class Phonon:
    """
    Attributes:
    -path
    -modelist

    -basis (angstrom)
    -atomic positions
    --cartesian (angstrom)
    --direct
    -elements
    -masses (amu)
    -qpoint
    -eigenvectors
    -eigenfreqs (cm^-1)

    -spacegroup
    -pointgroup
    -Raman tensors
    -dielectric tensor

    -Raman selection rules
    -IR selection rules

    -labels
    -acoustics
    -degenerates
    """ 
    def __init__(
        self,
        path: str = "",
        code_in: str = "phonopy",
        modelist: NDArray[int] = None,
        nosym: bool = False
    ) -> None:

        self.path = path

        if code_in == "phonopy":
            from parserPhonopy import parsePhonopy
            eigvals, eigvecs, _norms, qpoint, basis, _nat, elements, positions, masses = parsePhonopy(path, None)
            if modelist == None:
                modelist = range(1, 3*_nat+1)
            #
        elif code_in == "VASP":
            from parserVASP import getCellVASP, getModesVASP
            _nat, basis, positions, elements = getCellVASP(self.path+"POSCAR")
            if modelist == None:
                modelist = range(1, 3*_nat + 1)
            #
            eigvals, eigvecs, _norms = getModesVASP(self.path+"OUTCAR", modelist, _nat)
            qpoint = [0, 0, 0]
            masses = [periodTableMasses[element] for element in elements]
        else:
            print("__init__ of Phonon class failed: code \""+code_in+"\" not supported")
            sys.exit(1)
        #
        self._nat = _nat
        self._norms = _norms
        self.eigenfreqs = [eigvals[i-1] for i in modelist]
        self.eigenvecs = [eigvecs[i-1] for i in modelist]
        self.qpoint = qpoint
        self.basis = basis
        self.cartesian = positions
        self.elements = elements
        self.masses = masses
        self.modelist = modelist

        self.direct = np.empty((self._nat, 3))
        for atom in range(self._nat):
            self.direct[atom, :] = np.dot(np.linalg.inv(self.basis.T), self.cartesian[atom, :])
        #
        self._dataset = get_symmetry_dataset((self.basis, self.direct, [periodTable[element] for element in self.elements]), symprec=1.e-5)
        self.spacegroup = str(self._dataset["number"])
        self.pointgroup = str(self._dataset["pointgroup"])   
        self.ramantensors = analyzeRamanTensors(self.pointgroup, varprint=False)
        self.dielectrictensor = analyzeDielectricTensor(self.pointgroup, varprint=False)
                
        if nosym == False:
            self._labels = getIrrepsSymbols(path, self.basis, self.direct, self.elements, self.pointgroup)
            self.labels = [self._labels[i-1] for i in self.modelist]
            self.acoustics = getAcoustics(self.eigenvecs, self.eigenfreqs, self.masses)
            self.degenerates = getDegenerates(self.eigenfreqs, self.labels, prec=1e0)
            self.silent = getSilent(modelist, self._labels, self.pointgroup)
        else:
            self._labels = []
            self.labels = []
            self.acoustics = []
            self.degenerates = []
            self.silent = []
        #
    #

    def print_ramantensors(self):
        analyzeRamanTensors(self.pointgroup, varprint=True)
    #
    def print_dielectrictensor(self):
        analyzeDielectricTensor(self.pointgroup, varprint=True) 
    #
    def print_ramanselection(self):
        RamanSelection(self.pointgroup, self.ramantensors)
    #
    def print_irselection(self):
        IRSelection(self.pointgroup)
    #
    def displace(self, stepsize=0.001, code_out="VASP", scffile="scf.in"):
        displace(self.path, self.modelist, stepsize, code_out, self.eigenvecs, self._norms, self.basis, self._nat, self.elements, self.cartesian, scffile)
#

class Spectrum:
    """
    Attributes:
    -smearing
    -IR spectrum (all components)
    -Raman spectrum (all components)
    """

#