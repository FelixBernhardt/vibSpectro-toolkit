#!/usr/bin/env python

#
# the wrapper for the parsers, using ase wherever possible
#

from ase.io import read
from ase import Atoms
import numpy as np

class CalculatorParser:
    def __init__(self, path, filename, modelist=None):
        self.path = path
        self.filename = filename
        self.modelist = modelist

    def parse_structure(self):
        raise NotImplementedError

    def parse_basis(self):
        return None  # many codes do not expose basis

    def parse_vibrations(self):
        return None, None
    
    def parse_eps_inf(self):
        return None
    
    def parse_born_charges(self):
        return None
    #
#

# Own format
from src.IO import loadAtomsData, loadPhononsData, loadBornData, loadEpsInfData

class OwnParser(CalculatorParser):
    def parse_structure(self):
        symbols, positions, cell = loadAtomsData(self.filename)
        atoms = Atoms(symbols=symbols, positions=positions, cell=cell)
        return atoms

    def parse_vibrations(self):
        freqs, modes, modelist = loadPhononsData(self.filename)
        return freqs, modes
    
    def parse_eps_inf(self):
        eps_inf = loadEpsInfData(self.filename)
        return eps_inf

    def parse_born_charges(self):
        born = loadBornData(self.filename)
        return born
    #
#

# VASP
from src.parser.parserVASP import getModesVASP, getBornVASP, getEpsInfVASP, getOpticsVASP, writePOSCAR, linkVASP

class VASPParser(CalculatorParser):
    def parse_structure(self):
        atoms = read(self.filename)
        return atoms

    def parse_vibrations(self):
        try:
            atoms = read(self.filename)
            nat = len(atoms)
            if np.all(self.modelist == None):
                modelist = range(1,3*nat+1)
            else:
                if any(self.modelist > 3*nat):
                    modelist = range(1,3*nat+1)
                else:
                    modelist = self.modelist
            #
            freqs, modes, norms = getModesVASP(self.filename, modelist, nat)
            return freqs, modes      
        except IOError:
            return None, None

    def parse_eps_inf(self):
        try:
            epsInf = getEpsInfVASP(self.filename)
            return epsInf
        except Exception:
            return np.zeros((3, 3))

    def parse_born_charges(self):
        try:
            atoms = read(self.filename)
            nat = len(atoms)
            born = getBornVASP(self.filename, nat)
            return born
        except Exception:
            atoms = read(self.filename)
            nat = len(atoms)
            return np.zeros((nat, 3, 3))
    
    def parse_epsilon(self, mode, disp):
        return getOpticsVASP(self.path+"displacements/mode"+str(mode)+"_"+str(disp)+"/vasprun.xml")
    
    def write_file(self, nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, filename):
        return writePOSCAR(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm)
    
    def link_file(self, file):
        return linkVASP(file)
    #
#

# QE
from src.parser.parserQE import getOpticsQE, getModesQE, getEpsInfQE, getBornQE, linkQE, writeSCF

class QEParser(CalculatorParser):
    def parse_structure(self):
        return read(self.filename)

    def parse_vibrations(self):
        return getModesQE(self.path)
    
    def parse_epsilon(self, mode, disp):
        return getOpticsQE(self.path+"displacements/mode"+str(mode)+"_"+str(disp))
    
    def parse_eps_inf(self):
        try:
            epsInf = getEpsInfQE(self.path)
            return epsInf
        except Exception:
            return np.zeros((3, 3))

    def parse_born_charges(self):
        try:
            atoms = read(self.filename)
            nat = len(atoms)
            born = getBornQE(self.path, nat)
            return born
        except Exception:
            atoms = read(self.filename)
            nat = len(atoms)
            return np.zeros((nat, 3, 3))
    
    def write_file(self, nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, scffile):
        return writeSCF(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, scffile)
    
    def link_file(self, file):
        return linkQE(file)
    #
#

# phonopy
from src.parser.parserPhonopy import parsePhonopy
from phonopy.interface.phonopy_yaml import PhonopyYaml

class PhonopyParser(CalculatorParser):
    def parse_structure(self):
        frequencies, eigvecs, norms, qpoint, basis, nat, elements, cPos, masses = parsePhonopy(self.filename, None)

        cell = basis
        symbols = elements
        positions = cPos

        return Atoms(symbols=symbols, positions=positions, cell=cell)

    def parse_vibrations(self):
        frequencies, eigvecs, norms, qpoint, basis, nat, elements, cPos, masses = parsePhonopy(self.filename, None)

        return np.array(frequencies), np.array(eigvecs)
    
    def parse_eps_inf(self):
        yaml = PhonopyYaml().read(self.filename)
        epsilon = yaml.nac_params["dielectric"]
        
        return epsilon
    
    def parse_born_charges(self):
        yaml = PhonopyYaml().read(self.filename)
        born = yaml.nac_params["born"]
        
        return born
    #
#


# wrapper code

class ASEParser:
    def __init__(self, path, file, modelist=None, code_out="VASP"):
        self.file = file
        self.path = path
        self.filename = self.path + self.file
        self.modelist = modelist
        self.code_out = code_out
        self.backend = self.detect_backend()
        self.code = self.detect_code()

    def detect_backend(self):
        fn = self.file.lower()

        if "poscar" in fn or "contcar" in fn or "outcar" in fn:
            return VASPParser(self.path, self.filename, self.modelist)

        if fn.endswith(".pwo") or fn.endswith(".out") or "qe" in fn:
            return QEParser(self.path, self.filename)

        if "phonopy.yaml" in fn:
            return PhonopyParser(self.path, self.filename)
        
        if ".yaml" in fn:
            return OwnParser(self.path, self.filename)

        raise ValueError("Unknown calculator format")
    
    def detect_code(self):
        fn = self.code_out.lower()

        if "vasp" in fn:
            return VASPParser(self.path, self.filename, self.modelist)

        if "qe" in fn:
            return QEParser(self.path, self.filename)

        raise ValueError("Unknown calculator format")

    def get_structure(self):
        return self.backend.parse_structure()

    def get_basis(self):
        return self.backend.parse_basis()

    def get_vibrations(self):
        return self.backend.parse_vibrations()
    
    def get_born_charges(self):
        return self.backend.parse_born_charges()
    
    def get_epsilon_inf(self):
        return self.backend.parse_eps_inf()
    
    def get_epsilon(self, mode, disp):
        return self.code.parse_epsilon(mode, disp)
    
    def write_file(self, nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, filename):
        return self.code.write_file(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm, filename)
            
    def link_file(self, file):
        return self.code.link_file(file)

    #
#