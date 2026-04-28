from ase.io import read
from ase.vibrations import Vibrations

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

# VASP
from ase.io import read
from parserVASP import getModesVASP, getBornVASP, getEpsInfVASP

class VASPParser(CalculatorParser):
    def parse_structure(self):
        atoms = read(self.filename)
        return atoms

    def parse_vibrations(self):
        print(self.filename)
        try:
            atoms = read(self.filename)
            nat = len(atoms)
            freqs, modes, norms = getModesVASP(self.filename, self.modelist, nat)
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
import numpy as np
from ase.io import read

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
import yaml
import numpy as np

class PhonopyParser(CalculatorParser):
    def parse_structure(self):
        with open("phonopy.yaml") as f:
            data = yaml.safe_load(f)

        from ase import Atoms
        cell = data["unit_cell"]["lattice"]
        symbols = [a["symbol"] for a in data["unit_cell"]["points"]]
        positions = [a["coordinates"] for a in data["unit_cell"]["points"]]

        return Atoms(symbols=symbols, positions=positions, cell=cell)

    def parse_vibrations(self):
        with open("phonopy.yaml") as f:
            data = yaml.safe_load(f)

        freqs = []
        modes = []

        for qpt in data["phonon"]:
            for band in qpt["band"]:
                freqs.append(band["frequency"])
                modes.append(np.array(band["eigenvector"]))

        return np.array(freqs), modes
    #
#


# wrapper code
import os

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

        if "phonopy.yaml" or "qpoints.yaml" in fn:
            return PhonopyParser(self.filename)

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