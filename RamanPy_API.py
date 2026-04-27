#!/usr/bin/env python

#
# contains the phonon class storing all needed information
#
import numpy as np
from numpy.typing import NDArray
from spglib import get_symmetry_dataset
from RamanLib import periodTable, periodTableMasses, getAcoustics, getDegenerates, getSilent, analyzeDielectricTensor, analyzeRamanTensors, RamanSelection, IRSelection, getIrrepsSymbols, getPointgroup_pymole, getIrrepsSymbols_pymole, getBorn, getEpsInf
from IR import calcIR
from displace import calcdisplace
from calcTensors import calcTensors
from calcSpectrum import calcSpectrum
from plotSpectrum import plotSpectrum, plotIRspectrum

class Phonon:
    """
    Attributes:
    _nat -> the number of atoms in the unit cell
    _modelist -> the number of expected phonon modes, 3*_nat
    _labels_tmp -> the labels for all modes in _modelist, as output by phonopy (frequencies always in ascending order)
    _dataset -> the spglib object used to extract the symmetry configuration
    _norms -> euclidean norms of the phonon eigenvectors

    path -> the folder path where the calculated phonon eigenmodes can be found, and where the subsequent calculations are run, has to end with "/"
    modelist -> the modes the user wants to calculate, the ordering of modes is the same as used in the "code_in" software
    molecule -> is the structure to calculate a solid (default) or a molecule?
    
    code_in -> software to read the phonon information from
    ordering -> are the phonons ordered by ascending/descending frequency?
    code_out -> software to be used to calculate the Raman spectra and read the dielectric function from
    born -> code to read the effective charges from
    eps_inf -> code to read the "high frequency" dielectric function from

    basis -> basis vectors of unit cell in angstrom
    cartesian -> positions of ions in cartesian coords. (angstrom)
    direct -> position of ions in direct coords.
    elements -> element identifier for the ions
    masses -> atomic masses of the ions in amu
    eigenvectors -> phononic eigenvectors that have been found in "code_in", displacements in cartesian coords. and angstrom
    eigenfreqs -> phononic eigenfrequencies in cm^-1

    nosym -> do we want to ignore symmetries? Symmetries require a FORCE_CONSTANTS file, as well as all 3*_nat phonon modes to be present in "code_in"
    pointgroup -> pointgroup of the unit cell
    ramantensors -> general raman tensor associated with the point group
    dielectrictensor -> general dielectric tensor associated with the point group
    labels -> symetry labels of the phonon modes

    acoustic -> indizes of acoustic phonon modes
    silent -> indices of raman silent phonon modes
    degenerates -> tuple of indices for degenerate phonon modes

    stepsize -> scaling factor for the finite-differences method used to displace the ions along the phononic eigenvectors
    smearing -> the smearing to be applied to the spectra in cm⁻1
    temperature -> the temperature to calculate the specrta for in Kelvin
    photon_freq -> the photon energy of the laser light used to simulate the Raman spectra in eV

    ALL LO STUFF NOT IMPLEMENTED!
    qdir -> the momentum direction of the incoming photon in cartesian coordinates. This defines the outermost values in Porto's notation. Make sure to correctly account for LO modes!
    LOcorr -> the LO correction that needs to be applied for specific q-directions
    """

    def __init__(
        self,
        path: str = "",
        code_in: str = "phonopy",
        code_out: str = "VASP",
        modelist: NDArray[int] = np.array([0]),
        molecule: bool = False,
        nosym: bool = False,
        born: str = "", # change to code-in !! when implemented...
        eps_inf: str = "",
        stepsize: float = 0.001,
        smearing: float = 5.0,
        temperature: float = 300,
        photon_freq: float = 2.0,
        qdir: tuple = (1, 0, 0),
        LOcorr: NDArray[int] = np.array([0]),
    ) -> None:

        self.path = path

        if code_in == "phonopy":
            from parserPhonopy import parsePhonopy
            eigvals, eigvecs, _norms, qpoint, basis, _nat, elements, positions, masses = parsePhonopy(path, None)
            if np.all(modelist == 0):
                modelist = np.array(range(1, 3*_nat+1))
            #
            self.ordering = "ascending"
        elif code_in == "VASP":

            from parserVASP import getCellVASP, getModesVASP
            _nat, basis, positions, elements = getCellVASP(self.path+"POSCAR") # rewrite the parser such that it yields the _modelist
            if np.all(modelist == 0):
                modelist = np.array(range(1, 3*_nat + 1))
            else:
                nosym = True
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
        self.eigenfreqs = eigvals
        self.eigenvecs = eigvecs
        self.basis = basis
        self.cartesian = positions
        self.elements = elements
        self.masses = masses
        self.molecule = molecule
        self._modelist = range(1,3*self._nat+1)

        self.direct = np.empty((self._nat, 3))
        for atom in range(self._nat):
            self.direct[atom, :] = np.dot(np.linalg.inv(self.basis.T), self.cartesian[atom, :])
        #
        self.acoustics = getAcoustics(self.eigenvecs, self.eigenfreqs, self.masses)
        self.modelist = [mode for mode in modelist if mode not in self.acoustics]
                         
        if nosym == False:
            self.set_symmetries(modelist)
        else:
            self._dataset = []
            self.pointgroup = "",
            self.ramantensors = [],
            self.dielectrictensor = [],
            self._labels_tmp = []
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
        self.qdir = qdir
        self.LOcorr = LOcorr
        self.stepsize = stepsize
    #
    def set_symmetries(self, modelist):
        if self.molecule == False:
            self._dataset = get_symmetry_dataset((self.basis, self.direct, [periodTable[element] for element in self.elements]), symprec=1.e-5)
            self.pointgroup = str(self._dataset["pointgroup"])   
            self.ramantensors = analyzeRamanTensors(self.pointgroup, varprint=False)
            self.dielectrictensor = analyzeDielectricTensor(self.pointgroup, varprint=False)
            self._labels_tmp = getIrrepsSymbols(self.path, self.basis, self.direct, self.elements, self.pointgroup)
        else:
            self._dataset = ""
            self.pointgroup = getPointgroup_pymole(self.cartesian, self.elements)
            self.ramantensors = analyzeRamanTensors(self.pointgroup, varprint=False)
            self.dielectrictensor = analyzeDielectricTensor(self.pointgroup, varprint=False)
            self._labels_tmp = getIrrepsSymbols_pymole(self.eigenvecs, self.cartesian, self.elements, self.pointgroup)
        #
        if self.ordering == "ascending":
            self.labels = [self._labels_tmp[i-1] for i in self._modelist]
        elif self.ordering == "descending":
            self.labels = [self._labels_tmp[3*self._nat-i] for i in self._modelist]
        #    
        self.degenerates = getDegenerates(self.eigenfreqs, self.labels, prec=1e0)
        self.silent = getSilent(self._modelist, self.labels, self.pointgroup)
        self.modelist = [mode for mode in modelist if mode not in self.silent and mode not in self.acoustics]

        #self.modelist = [mode for mode in modelist if mode not in self.silent and mode not in self.acoustics and mode not in [x[1] for x in self.degenerates]]
        
        #if self.modelist != modelist:
        #    print("[__init__]: Removing Raman silent and degenerate modes")
        #
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
            calcIR(self.path, self.modelist, self.degenerates, self.silent, self.acoustics, self.eigenfreqs, self.eigenvecs, self.basis, self._nat, self.masses, self.born, self.smearing)
        #
    #
    def plotIR(self, lualatex=False):
        plotIRspectrum(self.path+"IR.dat", self.path, lualatex)
    #
    def displace(self, scffile="scf.in"):
        calcdisplace(self.path, self.modelist, self.stepsize, self.code_out, self.eigenvecs, self._norms, self.basis, self._nat, self.elements, self.cartesian, scffile)
    #
    def tensors(self):
        calcTensors(self.path, self.modelist, self.code_out, self.eigenfreqs, self._norms, self.basis, self.degenerates, self.labels, self.ramantensors, self.stepsize)
    #
    def spectrum(self):
        calcSpectrum(self.path, self.modelist, self.degenerates, self.acoustics, self.eigenfreqs, self.eigenvecs, self.basis, self._nat, self.born, self.eps_inf, self.photon_freq, self.temperature, self.smearing, self.qdir, self.LOcorr)        
    #
    def plotRaman(self, porto=["xx", "yy", "zz", "xy", "yz", "xz", "perp", "back"], lualatex=False):
        print("[plotRaman]: Plotting Raman spectrum")
        for pt in porto:
            plotSpectrum(self.path, self.photon_freq, pt, self.qdir, lualatex)
        #
        print("[plotRaman]: Done.") 
    #
#