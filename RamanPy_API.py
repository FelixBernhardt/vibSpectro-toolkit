#!/usr/bin/env python

#
# contains the phonon class storing all needed information
#
import os
import numpy as np
from numpy.typing import NDArray
from spglib import get_symmetry_dataset
from Symmetries import periodTable, getAcoustics, getRotations, getDegenerates, getDecomposition, getRamanSilent, analyzeDielectricTensor, analyzeRamanTensors, RamanSelection, IRSelection, getIrrepsSymbols
from IO import writeData, writeRaman, writeRamanSpectrum, writeConstantRaman, writeIRSpectrum, writeReflectanceSpectrum, loadSymmetryData, loadPhononsData, loadRamanTensor, loadConstantRaman, loadSpectrum, loadIR, loadReflectance
from IR import calcIR, calcReflectance
from LoTo import getLOFreqs
from displace import calcDisplace
from calcTensors import calcTensors
from calcSpectrum import calcSpectrum
from plotSpectrum import plotSpectrum, plotIRspectrum, plotRspectrum
from parserASE import ASEParser

class Phonon:
    """
    Attributes:
    _nat -> the number of atoms in the unit cell
    _modelist -> the number of expected phonon modes, 3*_nat
    _labels_tmp -> the labels for all modes in _modelist, as output by phonopy
    _dataset -> the spglib object used to extract the symmetry configuration
    _norms -> euclidean norms of the phonon eigenvectors, should be 1 for all modes

    path -> the folder path where the calculated phonon eigenmodes can be found, and where the subsequent calculations are run
    modelist -> the modes the user wants to calculate, the ordering of modes is the same as used in the "file" software
    
    ordering -> are the phonons ordered by ascending/descending frequency?
    code_out -> software to be used to calculate the Raman spectra
    born -> do we need effective charges?

    basis -> basis vectors of unit cell in angstrom
    cartesian -> positions of ions in cartesian coords. (angstrom)
    direct -> position of ions in direct coords.
    elements -> element identifier for the ions
    masses -> atomic masses of the ions in amu
    eigenvectors -> phononic eigenvectors that have been found in "file", displacements in cartesian coords. and angstrom, mass weighted
    eigenfreqs -> phononic eigenfrequencies in cm^-1

    nosym -> do we want to ignore symmetries? Symmetries require a FORCE_CONSTANTS file, as well as all 3*_nat phonon modes to be present in "file"
    pointgroup -> pointgroup of the unit cell
    ramantensors -> general raman tensor associated with the point group
    dielectrictensor -> general dielectric tensor associated with the point group
    labels -> symetry labels of the phonon modes
    decomposition -> phonon decomposition at Gamma point

    acoustic -> indices of acoustic phonon modes
    rotations -> indices of (almost) pure rotational modes, only relevant for isolated molecules
    silent -> indices of raman silent phonon modes
    degenerates -> tuple of indices for degenerate phonon modes

    stepsize -> scaling factor for the finite-differences method used to displace the ions along the phononic eigenvectors
    smearing -> the smearing to be applied to the spectra in cm⁻1
    temperature -> the temperature to calculate the specrta for in Kelvin
    photon_freq -> the photon energy of the laser light used to simulate the Raman spectra in eV

    ALL LO STUFF NOT IMPLEMENTED!
    qdir -> the momentum direction of the incoming photon in cartesian coordinates. This defines the outermost values in Porto's notation. Make sure to correctly account for LO modes!
    LOcorr -> do we need to correct for LO modes in geometry setup with qdir?

    version -> the version number
    """

    def __init__(
        self,
        name: str = "MySystem",
        path: str = "./",
        file: str = "phonopy.yaml",
        code_out: str = "VASP",
        modelist: NDArray[int] = None,
        nosym: bool = False,
        molecule: bool = False,
        stepsize: float = 0.01,
        smearing: float = 5.0,
        temperature: float = 300,
        stokes: str = "stokes",
        photon_freq: float = 2.0,
        born: bool = False,
        qdir: tuple = (1, 0, 0),
        LOcorr: bool = False,
    ) -> None:

        self.version = "0.0.1"

        ###############
        # general setup
        ###############
        self.name = name
        if path.endswith("/"):
            self.path = path
        else:
            self.path = path + "/"
        #
        self.file = file

        # just in case
        modelist = np.array(modelist)

        # unit cell and phonon mode information from external software
        parser = ASEParser(self.path+self.file, modelist=modelist)
        
        atoms = parser.get_structure()
        self._nat = len(atoms)
               
        if parser.backend.__class__.__name__ == "OwnParser":
            _, _, calcmodes = loadPhononsData(self.path+self.file)
            self._modelist = calcmodes
        else:
            self._modelist = np.array(range(1,3*self._nat+1))
        
        if np.all(modelist == None):
            modelist = self._modelist
        if any(mode not in self._modelist for mode in modelist):
            print("[__init__]: WARNING: not all requested modes are present in "+self.path+self.file+", continuing...")
            modelist = [mode for mode in modelist if mode in self._modelist]
       
        eigenfreqs, eigenvecs = parser.get_vibrations()
        self.elements = atoms.get_chemical_symbols()
        self.cartesian = atoms.get_positions()
        self.direct = atoms.get_scaled_positions()
        self.basis = atoms.get_cell()
        self.masses = atoms.get_masses()

        # phonopy uses q-direction in reciprocal direct coords, input qdir is assumed to be cartesian
        if qdir != (1,0,0) and qdir != (0,1,0) and qdir != (0,0,1) and qdir != (1,1,0) and qdir != (1,0,1) and qdir != (0,1,1) and qdir != (1,1,1):
            print("[__init__]: Invalid q-direction specified, resorting to default.")
            qdir = (1,0,0)
        #
        self.qdir_direct = np.linalg.solve(atoms.cell.reciprocal().T, np.array(qdir))
        self.qdir_cartesian = qdir

        # check the mode's ordering
        if np.all(np.diff(eigenfreqs) >= 0):
            self.ordering = "ascending"
        elif np.all(np.diff(eigenfreqs) <= 0):
            self.ordering = "descending"
        else:
            print("[__init__]: Could not detect ordering of frequencies !?")
        #
        self.eigenvecs = dict(zip(modelist, [eigenvecs[mode-1] for mode in modelist]))
        self.eigenfreqs = dict(zip(modelist, [eigenfreqs[mode-1] for mode in modelist]))
        self._norms = dict(zip(modelist, np.array([np.linalg.norm(self.eigenvecs[mode]) for mode in modelist])))

        # check for pure translations and rotations
        self.acoustics = getAcoustics(modelist, self.eigenvecs, self.eigenfreqs, self.masses)
        self.molecule = molecule
        if self.molecule == True:
            self.rotations = getRotations(modelist, self.masses, self.cartesian, self.eigenvecs)
        else:
            self.rotations = []
        #
        self.modelist = np.array([mode for mode in modelist if mode not in self.acoustics and mode not in self.rotations], dtype=int)

        # symmetry analysis of modes and pointgroup
        if nosym == False:
            if os.path.isfile(self.path+"FORCE_CONSTANTS") and parser.backend.__class__.__name__ != "OwnParser":
                self._dataset = get_symmetry_dataset((self.basis, self.direct, [periodTable[element] for element in self.elements]), symprec=1.e-5)
                #self.pointgroup = str(self._dataset.pointgroup)
                self.pointgroup = str(self._dataset["pointgroup"])
                self._labels_tmp = getIrrepsSymbols(self.path, self.basis, self.direct, self.elements, self.pointgroup)
                #
                if self.ordering == "ascending":
                    self.labels = dict(enumerate([self._labels_tmp[i-1] for i in self._modelist], start=1))
                elif self.ordering == "descending":
                    self.labels = dict(enumerate([self._labels_tmp[3*self._nat-i] for i in self._modelist], start=1))
                #
                self.set_symmetries(modelist)
            elif parser.backend.__class__.__name__ == "OwnParser":
                system, symmetry, pointgroup, labels = loadSymmetryData(self.path+self.file)
                self.name = system
                if symmetry == True:
                    nosym = False
                    self.pointgroup = pointgroup
                    self.labels = dict(zip(self._modelist, labels))
                    self.set_symmetries(modelist)
                else:
                    nosym = True
            else:
                print("[__init__]: Could not find FORCE_CONSTANTS in "+self.path+". Symmetry analysis is disabled.")
                nosym = True
            #
        #
        if nosym == True:
            print("[__init__]: Symmetry analysis is disabled.")
            #
            self._dataset = []
            self.pointgroup = ""
            self.ramantensors = []
            self.dielectrictensor = []
            self._labels_tmp = []
            self.labels = dict(zip(modelist, ["A1" for x in range(len(modelist))]))
            self.degenerates = []
            self.silent = []
            self.IRmodelist = self.modelist
            self.Ramanmodelist = self.modelist
        #
        if LOcorr == True or born == True:
            self.born = parser.get_born_charges()
            self.eps_inf = parser.get_epsilon_inf()
        else:
            self.born = np.zeros((self._nat, 3, 3))
            self.eps_inf = np.zeros((3, 3))
        #

        if code_out != "QE" and code_out != "VASP":
            print("[__init__]: "+code_out+" is not supported, some functionalities might fail. Continuing...")
        self.code_out = code_out
        
        self.smearing = smearing
        self.temperature = temperature
        self.photon_freq = photon_freq
        self.stokes = stokes
        self.LOcorr = LOcorr
        self.stepsize = stepsize

        # LO not implemented
        if self.LOcorr == True:
            print("[__init__]: LO correction not implemented, please switch off! Results may be unreliable")
        #
    #

    ###########################
    # Symmetry related fuctions
    ###########################
    def set_symmetries(self, modelist):
        self.ramantensors = analyzeRamanTensors(self.pointgroup, varprint=False)
        self.dielectrictensor = analyzeDielectricTensor(self.pointgroup, varprint=False)    
        self.degenerates = getDegenerates(self.modelist, self.eigenfreqs, self.labels, prec=1e0)
        self.silent = getRamanSilent(self._modelist, self.labels, self.pointgroup)
        #self.modelist = [mode for mode in modelist if mode not in self.silent and mode not in self.acoustics and mode not in self.rotations]
        self.IRmodelist = np.array([mode for mode in modelist if mode not in self.acoustics and mode not in self.rotations], dtype=int)
        self.Ramanmodelist = np.array([mode for mode in modelist if mode not in self.silent and mode not in self.acoustics and mode not in self.rotations], dtype=int)
        self.modelist = np.array([mode for mode in modelist if mode not in self.silent and mode not in self.acoustics and mode not in self.rotations and mode not in [x[1] for x in self.degenerates]], dtype=int)
        #
    #
    def print_decomposition(self):
        if self.pointgroup == "":
            print("[print_ramantensors]: ERROR, need pointgroup")
        else:
            labellist = [self.labels[mode] for mode in self._modelist if mode not in self.acoustics and mode not in self.rotations]
            getDecomposition(labellist)
        #
    #
    def print_ramantensors(self):
        if self.pointgroup == "":
            print("[print_ramantensors]: ERROR, need pointgroup")
        else:
            analyzeRamanTensors(self.pointgroup, varprint=True)
        #
    #
    def print_dielectrictensor(self):
        if self.pointgroup == "":
            print("[print_dielectritensor]: ERROR, need pointgroup")
        else:
            analyzeDielectricTensor(self.pointgroup, varprint=True)
        # 
    #
    def print_ramanselection(self):
        if self.pointgroup == "" or self.ramantensors == []:
            print("[print_ramanselection]: ERROR, need pointgroup and corresponding general Ramantensors")
        else:
            RamanSelection(self.pointgroup, self.ramantensors)
        #
    #
    def print_irselection(self):
        if self.pointgroup == "":
            print("[print_irselection]: ERROR, need pointgroup")
        else:
            IRSelection(self.pointgroup)
        #
    #

    ########################
    # Infrared spectroscopy
    ########################
    def calc_ir(self):
        parser = ASEParser(self.path+self.file, modelist=self.modelist)
        self.born = parser.get_born_charges()
        if np.all(self.born == 0):
            print("[IR]: ERROR, need effective charges")
        else:
            self.IR_data = calcIR(self.IRmodelist, self.eigenfreqs, self.eigenvecs, self.basis, self._nat, self.masses, self.born, self.smearing)
        #
    #
    def write_ir(self):
        writeIRSpectrum(self)
    #
    def plot_ir(self, lualatex=False):
        plotIRspectrum(self.IR_data, self.path, lualatex)
    #
    def calc_reflectance(self):
        if not hasattr(self, "IR_data"):
            parser = ASEParser(self.path+self.file, modelist=self.modelist)
            self.born = parser.get_born_charges()
            if np.all(self.born == 0):
                self.IR_data = calcIR(self.IRmodelist, self.eigenfreqs, self.eigenvecs, self.basis, self._nat, self.masses, self.born, self.smearing)
        #
        self.reflectance_data = calcReflectance(self.IR_data)
    #
    def write_reflectance(self):
        writeReflectanceSpectrum(self)
    #
    def plot_reflectance(self, lualatex=False):
        plotRspectrum(self.reflectance_data, self.path, lualatex)
    #

    ####################
    # Raman spectroscopy
    ####################
    def calc_raman_displace(self, scffile="scf.in"):
        calcDisplace(self.path, self.modelist, self.stepsize, self.code_out, self.eigenvecs, self._norms, self.basis, self._nat, self.elements, self.cartesian, scffile)
    #
    def calc_raman_tensors(self):
        # format [mode_index][[w, xx, yy, zz, xy, yz, xz, perp, back]]
        self.ramantensors_data = calcTensors(self.path, self.modelist, self.code_out, self.eigenfreqs, self._norms, self.basis, self.degenerates, self.labels, self.ramantensors, self.stepsize)
    #
    def write_raman_tensors(self):
        writeRaman(self)
    #
    def calc_raman_spectrum(self):
        # check for LO
        if self.LOcorr == True:
            self.eigenfreqs_LO = getLOFreqs(self.path, self.IRmodelist, self.qdir_cartesian, self.qdir_direct, self.ordering, self.eigenvecs, self.degenerates, self.labels)
        else:
            self.eigenfreqs_LO = self.eigenfreqs
        #
        self.constantraman_data, self.ramanspectrum_data = calcSpectrum(self.path, self.ramantensors_data, self.modelist, self.Ramanmodelist, self.eigenfreqs_LO, self.eigenvecs, self.basis, self._nat, self.born, self.eps_inf, self.photon_freq, self.temperature, self.smearing, self.stokes, self.qdir_cartesian, self.LOcorr)        
    #
    def write_raman_spectrum(self):
        writeConstantRaman(self)
        writeRamanSpectrum(self)
    #
    def plot_raman(self, porto=["xx", "yy", "zz", "xy", "yz", "xz", "perp", "back"], lualatex=False):
        print("[plot_Raman]: Plotting Raman spectrum")
        for pt in porto:
            plotSpectrum(self.ramanspectrum_data, self.path, self.photon_freq, pt, self.qdir, lualatex)
        #
        print("[plot_Raman]: Done.") 
    #

    ########
    # IO
    ########
    def write_system(self):
        writeData(self)
    #
    def load_raman_tensors(self, mode=1):
        self.ramantensors_data[mode] = loadRamanTensor(self.path, mode)
    #
    def load_raman_tensors_const(self, filename="Raman.yaml"):
        self.constantraman_data = loadConstantRaman(self.path+filename)
    #
    def load_raman_spectrum(self, filename="Intensity.yaml"):
        self.ramanspectrum_data, self.qdir = loadSpectrum(self.path+filename)
    #
    def load_ir(self, filename="IR.yaml"):
        self.IR_data = loadIR(self.path+filename)
    #
    def load_reflectance(self, filename="Reflectance.yaml"):
        self.reflectance_data = loadReflectance(self.path+filename)
    #
#