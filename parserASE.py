#!/usr/bin/env python

#
# the wrapper for the parsers, using ase wherever possible
#

from ase.io import read
from ase import Atoms
import numpy as np

class CalculatorParser:
    def __init__(self, filename, modelist=None):
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
from IO import loadAtomsData, loadPhononsData, loadBornData, loadEpsInfData
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
from parserVASP import getModesVASP, getBornVASP, getEpsInfVASP

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
    #
#

# QE
class QEParser(CalculatorParser):
    def parse_structure(self):
        return read(self.filename)

    def parse_vibrations(self):
        freqs = []
        modes = []

        with open("matdyn.modes") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            if "freq" in line.lower():
                parts = line.split()
                freq = float(parts[-2])  # cm^-1
                freqs.append(freq)

                mode = []
                i += 1
                while i < len(lines) and len(lines[i].split()) == 6:
                    x, y, z, dx, dy, dz = map(float, lines[i].split())
                    mode.append([dx, dy, dz])
                    i += 1
                modes.append(np.array(mode))
            else:
                i += 1

        return np.array(freqs), modes
    #
#

# phonopy
from parserPhonopy import parsePhonopy
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
    def __init__(self, filename, modelist=None):
        self.filename = filename
        self.modelist = modelist
        self.backend = self.detect_backend()

    def detect_backend(self):
        fn = self.filename.lower()

        if "poscar" in fn or "contcar" in fn or "outcar" in fn:
            return VASPParser(self.filename, self.modelist)

        if fn.endswith(".pwo") or fn.endswith(".out") or "qe" in fn:
            return QEParser(self.filename)

        if "phonopy.yaml" in fn:
            return PhonopyParser(self.filename)
        
        if ".yaml" in fn:
            return OwnParser(self.filename)

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
    #
#