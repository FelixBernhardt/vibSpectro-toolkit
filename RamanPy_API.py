#!/usr/bin/env python

#
# contains the phonon class storing all needed information
#
import numpy as np
from numpy.typing import NDArray
from RamanLib import periodTable, periodTableMasses, getAcoustics, getDegenerates, getSilent, getBorn, getEpsInf
from symmetries import *
from IR import calcIR
from displace import calcdisplace
from calcTensors import calcTensors
from calcSpectrum import calcSpectrum

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
        code_out: str = "VASP",
        modelist: NDArray[int] = None,
        nosym: bool = False,
        born: str = "", # change to code-in !! when implemented...
        eps_inf: str = "",
        stepsize: float = 0.001,
        smearing: float = 5.0,
        temperature: float = 300,
        photon_freq: float = 2.0,
        plot: bool = False,
        qdir: list = (1, 0, 0),
        LOcorr: NDArray[int] = None,
    ) -> None:

        self.path = path

        if code_in == "phonopy":
            from parserPhonopy import parsePhonopy
            eigvals, eigvecs, _norms, qpoint, basis, _nat, elements, positions, masses = parsePhonopy(path, None)
            if modelist == None:
                modelist = range(1, 3*_nat+1)
            #
            self.ordering = "ascending"
        elif code_in == "VASP":
            from parserVASP import getCellVASP, getModesVASP
            _nat, basis, positions, elements = getCellVASP(self.path+"POSCAR")
            if modelist == None:
                modelist = range(1, 3*_nat + 1)
            #
            eigvals, eigvecs, _norms = getModesVASP(self.path+"OUTCAR", modelist, _nat)
            masses = [periodTableMasses[element] for element in elements]
            self.ordering = "descending"
        else:
            print("__init__ of Phonon class failed: code \""+code_in+"\" not supported")
            return None
        #
        self._nat = _nat
        self._norms = _norms
        self.eigenfreqs = [eigvals[i-1] for i in modelist]
        self.eigenvecs = [eigvecs[i-1] for i in modelist]
        self.basis = basis
        self.cartesian = positions
        self.elements = elements
        self.masses = masses
        self._modelist = modelist

        self.direct = np.empty((self._nat, 3))
        for atom in range(self._nat):
            self.direct[atom, :] = np.dot(np.linalg.inv(self.basis.T), self.cartesian[atom, :])
        #
        self.acoustics = getAcoustics(self.eigenvecs, self.eigenfreqs, self.masses)
        self.modelist = [mode for mode in self._modelist if mode not in self.acoustics]
                         
        if nosym == False:
            self.set_symmetries()
        else:
            self._dataset = []
            self.spacegroup = "",
            self.pointgroup = "",
            self.ramantensors = [],
            self.dielectrictensor = [],
            self._labels = []
            self.labels = []
            self.degenerates = []
            self.silent = []
        #
        if born != "":
            self.born = getBorn(self.path, born, self._nat)
        else:
            self.born = np.zeros((self._nat, 3, 3))
        #
        if eps_inf != "":
            self.eps_inf = getEpsInf(self.path, eps_inf)
        else:
            self.eps_inf = np.zeros((3, 3))
        #

        self.code_out = code_out
        self.smearing = smearing
        self.temperature = temperature
        self.photon_freq = photon_freq
        self.plot = plot
        self.qdir = qdir
        self.LOcorr = LOcorr
        self.stepsize = stepsize
    #
    def set_symmetries(self):
        self._dataset = get_symmetry_dataset((self.basis, self.direct, [periodTable[element] for element in self.elements]), symprec=1.e-5)
        self.spacegroup = str(self._dataset["number"])
        self.pointgroup = str(self._dataset["pointgroup"])   
        self.ramantensors = analyzeRamanTensors(self.pointgroup, varprint=False)
        self.dielectrictensor = analyzeDielectricTensor(self.pointgroup, varprint=False)
        self._labels = getIrrepsSymbols(self.path, self.basis, self.direct, self.elements, self.pointgroup)
        if self.ordering == "ascending":
            self.labels = [self._labels[i-1] for i in self._modelist]
        elif self.ordering == "descending":
            self.labels = [self._labels[3*self._nat-i] for i in self._modelist]
        #    
        self.degenerates = getDegenerates(self.eigenfreqs, self.labels, prec=1e0)
        self.silent = getSilent(self.modelist, self.labels, self.pointgroup)
        self.modelist = [mode for mode in self._modelist if mode not in self.silent and mode not in self.acoustics and mode not in [x[1] for x in self.degenerates]]
    #
    def print_ramantensors(self):
        if self.pointgroup == "":
            print("ERROR, need pointgroup")
        else:
            analyzeRamanTensors(self.pointgroup, varprint=True)
        #
    #
    def print_dielectrictensor(self):
        if self.pointgroup == "":
            print("ERROR, need pointgroup")
        else:
            analyzeDielectricTensor(self.pointgroup, varprint=True)
        # 
    #
    def print_ramanselection(self):
        if self.pointgroup == "" or self.ramantensors == []:
            print("ERROR, need pointgroup and corresponding general Ramantensors")
        else:
            RamanSelection(self.pointgroup, self.ramantensors)
        #
    #
    def print_irselection(self):
        if self.pointgroup == "":
            print("ERROR, need pointgroup")
        else:
            IRSelection(self.pointgroup)
        #
    #
    def IR(self):
        if np.all(self.born == 0):
            print("ERROR, need effective charges")
        else:
            calcIR(self.path, self.modelist, self.degenerates, self.silent, self.acoustics, self.eigenfreqs, self.eigenvecs, self.basis, self._nat, self.masses, self.born, self.smearing, self.plot)
        #
    #
    def displace(self, scffile="scf.in"):
        calcdisplace(self.path, self.modelist, self.stepsize, self.code_out, self.eigenvecs, self._norms, self.basis, self._nat, self.elements, self.cartesian, scffile)
    #
    def tensors(self):
        calcTensors(self.path, self.modelist, self.code_out, self.eigenfreqs, self._norms, self.basis, self.degenerates, self.labels, self.ramantensors, self.stepsize)
    #
    def spectrum(self):
        calcSpectrum(self.path, self.modelist, self.degenerates, self.acoustics, self.eigenfreqs, self.eigenvecs, self.basis, self._nat, self.born, self.eps_inf, self.photon_freq, self.temperature, self.smearing, self.qdir, self.LOcorr, self.plot)        
    #
#