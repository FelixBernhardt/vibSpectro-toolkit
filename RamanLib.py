#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import sys
import numpy as np
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.irreps import IrReps
from parserVASP import getBornVASP

periodTable = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
   'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
   'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
   'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
   'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
   'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
       'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
   'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,'Re': 75,
   'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84,'At': 85, 'Rn': 86,
   'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97,
   'Cf': 98,'Es': 99, 'Fm': 100, 'Md': 101, 'No':102, 'Lr': 103}

# masses in atomic units
# from https://www.angelo.edu/faculty/kboudrea/periodic/structure_mass.htm
periodTableMasses = {'H':  1.00797, 'He':  4.00260, 'Li':  6.941,
         'Be':  9.01218, 'B':   10.81, 'C':     12.011,
         'N':   14.0067, 'O':   15.9994, 'F': 18.998403,
         'Ne':  20.179, 'Na':   22.98977, 'Mg': 24.305,
         'Al':  26.98154, 'Si': 28.0855, 'P':   30.97376,
         'S':   32.06, 'Cl':    35.453, 'K':    39.0983,
         'Ar':  39.948, 'Ca':   40.08, 'Sc':    44.9559,
         'Ti':  47.90, 'V':     50.9415, 'Cr':  51.996,
         'Mn':  54.9380, 'Fe':  55.847, 'Ni':   58.70,
         'Co':  58.9332, 'Cu':  63.546, 'Zn':   65.38,
         'Ga':  69.72, 'Ge':    72.59, 'As':    74.9216,
         'Se':  78.96, 'Br':    79.904, 'Kr':   83.80,
         'Rb':  85.4678, 'Sr':  87.62, 'Y':     88.9059,
         'Zr':  91.22, 'Nb':    92.9064, 'Mo':  95.94,
         'Tc':  98, 'Ru':       101.07, 'Rh':   102.9055,
         'Pd':  106.4, 'Ag':    107.868, 'Cd':  112.41,
         'In':  114.82, 'Sn':   118.69, 'Sb':   121.75,
         'I':   126.9045, 'Te': 127.60, 'Xe':   131.30,
         'Cs':  132.9054, 'Ba': 137.33, 'La':   138.9055,
         'Ce':  140.12, 'Pr':   140.9077, 'Nd': 144.24,
         'Pm':  145, 'Sm':      150.4, 'Eu':    151.96,
         'Gd':  157.25, 'Tb':   158.9254, 'Dy': 162.50,
         'Ho':  164.9304, 'Er': 167.26, 'Tm':   168.9342,
         'Yb':  173.04, 'Lu':   174.967, 'Hf':  178.49,
         'Ta':  180.9479, 'W':  183.85, 'Re':   186.207,
         'Os':  190.2, 'Ir': 192.22, 'Pt':      195.09,
         'Au':  196.9665, 'Hg': 200.59, 'Tl':   204.37,
         'Pb':  207.2, 'Bi':    208.9804, 'Po': 209,
         'At':  210, 'Rn':      222, 'Fr':      223,
         'Ra':  226.0254, 'Ac': 227.0278, 'Pa': 231.0359,
         'Th':  232.0381, 'Np': 237.0482, 'U':  238.029}


backDirs = ["x(yy)x\u0305", "x(yz)x\u0305", "x(zz)x\u0305", "y(xx)y\u0305", "y(xz)y\u0305", "y(zz)y\u0305", "z(xx)z\u0305", "z(xy)z\u0305", "z(yy)z\u0305"]
bdDir = {0: (1,1), 1: (1,2), 2: (2,2), 3: (0,0), 4: (0,2), 5: (2,2), 6: (0,0), 7: (0,1), 8: (1,1)}
rightDirs = ["x(yx)y", "x(yz)y", "x(zx)y", "x(zz)y", "x(yx)z", "x(yy)z", "x(zx)z", "x(zy)z", "y(xx)z", "y(xy)z", "y(zx)z", "y(zy)z"]
rDir = {0: (0,1), 1: (1,2), 2: (0,2), 3: (2,2), 4: (0,1), 5: (1,1), 6: (0,2), 7: (1,2), 8: (0,0), 9:(0,1), 10: (0,2), 11: (1,2)}
IRDirs = ["E || x", "E || y", "E || z"]
portoq = {(0, 0, 1) : "zz", (0, 1, 0) : "yy", (1, 0, 0) : "xx", (1, 1, 0) : "xy", (1, 0, 1): "xz", (0, 1, 1) : "yz" }

e_charge = 1.602176634e-19   # C
amu      = 1.66053906660e-27 # kg
eps0     = 8.8541878128e-12  # F/m
c_cm     = 2.99792458e10     # cm/s
h        = 6.62606957e-34    # Js
kb       = 1.3806488e-23     # J/K
ev2rcm   = 8065.5401

HM_TO_SCHOENFLIES = {
    "1":      "C1",
    "-1":     "Ci",
    "2":      "C2",
    "m":      "Cs",
    "2/m":    "C2h",
    "222":    "D2",
    "mm2":    "C2v",
    "mmm":    "D2h",
    "4":      "C4",
    "-4":     "S4",
    "4/m":    "C4h",
    "422":    "D4",
    "4mm":    "C4v",
    "-42m":   "D2d",
    "4/mmm":  "D4h",
    "3":      "C3",
    "-3":     "C3i",
    "32":     "D3",
    "3m":     "C3v",
    "-3m":    "D3d",
    "6":      "C6",
    "-6":     "C3h",
    "6/m":    "C6h",
    "622":    "D6",
    "6mm":    "C6v",
    "-6m2":   "D3h",
    "6/mmm":  "D6h",
    "23":     "T",
    "m-3":    "Th",
    "432":    "O",
    "-43m":   "Td",
    "m-3m":   "Oh"
}

POINTGROUP_TO_INT = {
    "C1":   1,
    "Ci":   2,   # = S2
    "C2":   3,
    "Cs":   4,   # = C1h
    "C2h":  5,
    "D2":   6,
    "C2v":  7,
    "D2h":  8,
    "C4":   9,
    "S4":   10,
    "C4h":  11,
    "D4":   12,
    "C4v":  13,
    "D2d":  14,
    "D4h":  15,
    "C3":   16,
    "C3i":  17,  # = S6
    "D3":   18,
    "C3v":  19,
    "D3d":  20,
    "C6":   21,
    "C3h":  22,
    "C6h":  23,
    "D6":   24,
    "C6v":  25,
    "D3h":  26,
    "D6h":  27,
    "T":    28,
    "Th":   29,
    "O":    30,
    "Td":   31,
    "Oh":   32
}

CHAR_TABLES = {
    # 1. Triclinic
    "C1": {
        "A": [1],
    },

    "Ci": {
        "Ag": [1, 1],
        "Au": [1, -1],
    },

    # 2. Monoclinic
    "C2": {
        "A": [1, 1],
        "B": [1, -1],
    },

    "Cs": {
        "A'": [1, 1],
        "A''": [1, -1],
    },

    "C2h": {
        "Ag": [1, 1, 1, 1],
        "Bg": [1, -1, 1, -1],
        "Au": [1, 1, -1, -1],
        "Bu": [1, -1, -1, 1],
    },

    # 3. Orthorhombic
    "D2": {
        "A": [1, 1, 1, 1],
        "B1": [1, 1, -1, -1],
        "B2": [1, -1, 1, -1],
        "B3": [1, -1, -1, 1],
    },

    "C2v": {
        "A1": [1, 1, 1, 1],
        "A2": [1, 1, -1, -1],
        "B1": [1, -1, 1, -1],
        "B2": [1, -1, -1, 1],
    },

    "D2h": {
        "Ag": [1, 1, 1, 1, 1, 1, 1, 1],
        "B1g": [1, 1, -1, -1, 1, 1, -1, -1],
        "B2g": [1, -1, 1, -1, 1, -1, 1, -1],
        "B3g": [1, -1, -1, 1, 1, -1, -1, 1],
        "Au": [1, 1, 1, 1, -1, -1, -1, -1],
        "B1u": [1, 1, -1, -1, -1, -1, 1, 1],
        "B2u": [1, -1, 1, -1, -1, 1, -1, 1],
        "B3u": [1, -1, -1, 1, -1, 1, 1, -1],
    },

    # 4. Tetragonal
    "C4": {
        "A": [1, 1, 1, 1],
        "B": [1, -1, 1, -1],
        "E": [2, 0, -2, 0],
    },

    "S4": {
        "A": [1, 1, 1, 1],
        "B": [1, -1, 1, -1],
        "E": [2, 0, -2, 0],
    },

    "C4h": {
        "Ag": [1, 1, 1, 1, 1, 1, 1, 1],
        "Bg": [1, -1, 1, -1, 1, -1, 1, -1],
        "Eg": [2, 0, -2, 0, 2, 0, -2, 0],
        "Au": [1, 1, 1, 1, -1, -1, -1, -1],
        "Bu": [1, -1, 1, -1, -1, 1, -1, 1],
        "Eu": [2, 0, -2, 0, -2, 0, 2, 0],
    },

    "C4v": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "B1": [1, -1, 1, 1, -1],
        "B2": [1, -1, 1, -1, 1],
        "E": [2, 0, -2, 0, 0],
    },

    "D4": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "B1": [1, -1, 1, 1, -1],
        "B2": [1, -1, 1, -1, 1],
        "E": [2, 0, -2, 0, 0],
    },

    "D2d": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "B1": [1, -1, 1, 1, -1],
        "B2": [1, -1, 1, -1, 1],
        "E": [2, 0, -2, 0, 0],
    },

    "D4h": {
        "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "A2g": [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
        "B1g": [1, -1, 1, 1, -1, 1, -1, 1, 1, -1],
        "B2g": [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
        "Eg": [2, 0, -2, 0, 0, 2, 0, -2, 0, 0],
        "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
        "A2u": [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
        "B1u": [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        "B2u": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        "Eu": [2, 0, -2, 0, 0, -2, 0, 2, 0, 0],
    },

    # 5. Trigonal
    "C3": {
        "A": [1, 1, 1],
        "E": [2, -1, -1],
    },

    "S6": {
        "Ag": [1, 1, 1, 1, 1, 1],
        "Au": [1, 1, 1, -1, -1, -1],
        "Eg": [2, -1, -1, 2, -1, -1],
        "Eu": [2, -1, -1, -2, 1, 1],
    },

    "C3v": {
        "A1": [1, 1, 1],
        "A2": [1, 1, -1],
        "E": [2, -1, 0],
    },

    "D3": {
        "A1": [1, 1, 1],
        "A2": [1, 1, -1],
        "E": [2, -1, 0],
    },

    "D3d": {
        "A1g": [1, 1, 1, 1, 1, 1],
        "A2g": [1, 1, -1, 1, 1, -1],
        "Eg": [2, -1, 0, 2, -1, 0],
        "A1u": [1, 1, 1, -1, -1, -1],
        "A2u": [1, 1, -1, -1, -1, 1],
        "Eu": [2, -1, 0, -2, 1, 0],
    },

    # 6. Hexagonal
    "C6": {
        "A": [1, 1, 1, 1, 1, 1],
        "B": [1, -1, 1, -1, 1, -1],
        "E1": [2, 1, -1, -2, -1, 1],
        "E2": [2, -1, -1, 2, -1, -1],
    },

    "C3h": {
        "A'": [1, 1, 1, 1, 1, 1],
        "A''": [1, 1, 1, -1, -1, -1],
        "E'": [2, -1, -1, 2, -1, -1],
        "E''": [2, -1, -1, -2, 1, 1],
    },

    "C6h": {
        "Ag": [1]*12,
        "Bg": [1, -1]*6,
        "E1g": [2, 1, -1, -2, -1, 1]*2,
        "E2g": [2, -1, -1, 2, -1, -1]*2,
        "Au": [1]*6 + [-1]*6,
        "Bu": [1, -1]*3 + [-1, 1]*3,
        "E1u": [2, 1, -1, -2, -1, 1] + [-2, -1, 1, 2, 1, -1],
        "E2u": [2, -1, -1, 2, -1, -1] + [-2, 1, 1, -2, 1, 1],
    },

    "C6v": {
        "A1": [1, 1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1, -1],
        "B1": [1, -1, 1, 1, -1, 1],
        "B2": [1, -1, 1, -1, 1, -1],
        "E1": [2, 1, -1, 0, -1, 1],
        "E2": [2, -1, -1, 0, -1, -1],
    },

    "D6": {
        "A1": [1, 1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1, -1],
        "B1": [1, -1, 1, 1, -1, 1],
        "B2": [1, -1, 1, -1, 1, -1],
        "E1": [2, 1, -1, 0, -1, 1],
        "E2": [2, -1, -1, 0, -1, -1],
    },

    "D3h": {
        "A1'": [1, 1, 1, 1, 1, 1],
        "A2'": [1, 1, -1, 1, 1, -1],
        "E'": [2, -1, 0, 2, -1, 0],
        "A1''": [1, 1, 1, -1, -1, -1],
        "A2''": [1, 1, -1, -1, -1, 1],
        "E''": [2, -1, 0, -2, 1, 0],
    },

    "D6h": {
        "A1g": [1]*12,
        "A2g": [1, 1, -1, 1, 1, -1]*2,
        "B1g": [1, -1, 1, 1, -1, 1]*2,
        "B2g": [1, -1, 1, -1, 1, -1]*2,
        "E1g": [2, 1, -1, 0, -1, 1]*2,
        "E2g": [2, -1, -1, 0, -1, -1]*2,
        "A1u": [1]*6 + [-1]*6,
        "A2u": [1, 1, -1, -1, -1, 1]*2,
        "B1u": [1, -1, 1, -1, 1, -1]*2,
        "B2u": [1, -1, 1, 1, -1, 1]*2,
        "E1u": [2, 1, -1, 0, -1, 1] + [-2, -1, 1, 0, 1, -1],
        "E2u": [2, -1, -1, 0, -1, -1] + [-2, 1, 1, 0, 1, 1],
    },

    # 7. Cubic
    "T": {
        "A": [1, 1, 1],
        "E": [2, -1, 2],
        "T": [3, 0, -1],
    },

    "Th": {
        "Ag": [1, 1, 1, 1, 1, 1],
        "Eg": [2, -1, 2, 2, -1, 2],
        "Tg": [3, 0, -1, 3, 0, -1],
        "Au": [1, 1, 1, -1, -1, -1],
        "Eu": [2, -1, 2, -2, 1, -2],
        "Tu": [3, 0, -1, -3, 0, 1],
    },

    "O": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "E": [2, -1, 2, 0, 0],
        "T1": [3, 0, -1, 1, -1],
        "T2": [3, 0, -1, -1, 1],
    },

    "Td": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "E": [2, -1, 2, 0, 0],
        "T1": [3, 0, -1, 1, -1],
        "T2": [3, 0, -1, -1, 1],
    },

    "Oh": {
        "A1g": [1]*10,
        "A2g": [1, 1, 1, -1, -1]*2,
        "Eg": [2, -1, 2, 0, 0]*2,
        "T1g": [3, 0, -1, 1, -1]*2,
        "T2g": [3, 0, -1, -1, 1]*2,
        "A1u": [1]*5 + [-1]*5,
        "A2u": [1, 1, 1, -1, -1] + [-1, -1, -1, 1, 1],
        "Eu": [2, -1, 2, 0, 0] + [-2, 1, -2, 0, 0],
        "T1u": [3, 0, -1, 1, -1] + [-3, 0, 1, -1, 1],
        "T2u": [3, 0, -1, -1, 1] + [-3, 0, 1, 1, -1],
    }
}

dielectricFunctionComponents = {
        "1": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),
       "-1": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),
    
        "2": np.array([["a", 0, "d"], [0, "b", 0], ["d", 0, "c"]], dtype = "str"),
        "m": np.array([["a", 0, "d"], [0, "b", 0], ["d", 0, "c"]], dtype = "str"),
    
      "2/m": np.array([["a", 0, "d"], [0, "b", 0], ["d", 0, "c"]], dtype = "str"),
      "222": np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype = "str"),
      "mm2": np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype = "str"),
      "mmm": np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype = "str"),

        "4": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),
       "-4": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),
      "4/m": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),
      "422": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),
      "4mm": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),
     "-42m": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),
    "4/mmm": np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str"),

        "3": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
       "-3": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
       "32": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
       "3m": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
      "-3m": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "6": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
       "-6": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
      "6/m": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
      "622": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
      "6mm": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
       "62": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
    "6/mmm": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),

       "23": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
      "m-3": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
      "432": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
     "-43m": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
     "m-3m": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str")
}

RamanTensorComponents = {
    "1": [
        "A", np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str")
    ],

    "-1": [
        "Ag", np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype = "str")
    ],

    "2": [
        "A", np.array([["a", "d", 0], ["d", "b", 0], [0, 0, "c"]], dtype = "str"),
        "B",  np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype = "str")
    ],

    "m": [
        "A'", np.array([["a", "d", 0], ["d", "b", 0], [0, 0, "c"]], dtype = "str"),
        "A''", np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype = "str")
    ],

    "2/m": [
        "Ag", np.array([["a", "d", 0], ["d", "b", 0], [0, 0, "c"]], dtype = "str"),
        "Bg", np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype = "str")
    ],

    "222": [
        "A", np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype = "str"),
        "B1", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str"),
        "B2", np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype = "str"),
        "B3", np.array([[0, 0, 0], [0, 0, "f"], [0, "f", 0]], dtype = "str")
    ],

    "mm2": [
        "A1", np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype = "str"),
        "A2", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str"),
        "B1", np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype = "str"),
        "B2", np.array([[0, 0, 0], [0, 0, "f"], [0, "f", 0]], dtype = "str")
    ],

    "mmm": [
        "Ag", np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype = "str"),
        "B1g", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str"),
        "B2g", np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype = "str"),
        "B3g", np.array([[0, 0, 0], [0, 0, "f"], [0, "f", 0]], dtype = "str")
    ],

    "4": [
        "A", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "B", np.array([["c", "d", 0], ["d", "-c", 0], [0, 0, 0]], dtype = "str"),
        "1E", np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype = "str"),
        "2E", np.array([[0, 0, "-f"], [0, 0, "e"], ["-f", "e", 0]], dtype = "str")
    ],

    "-4": [
        "A", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "B", np.array([["c", "d", 0], ["d", "-c", 0], [0, 0, 0]], dtype = "str"),
        "1E", np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype = "str"),
        "2E", np.array([[0, 0, "f"], [0, 0, "-e"], ["f", "-e", 0]], dtype = "str")
    ],

    "4/m": [
        "Ag", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "Bg", np.array([["c", "d", 0], ["d", "-c", 0], [0, 0, 0]], dtype = "str"),
        "1Eg", np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype = "str"),
        "2Eg", np.array([[0, 0, "-f"], [0, 0, "e"], ["-f", "e", 0]], dtype = "str")
    ],

    "4222": [
        "A", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "B1", np.array([["c", 0, 0], [0, "-c", 0], [0, 0, 0]], dtype = "str"),
        "B2", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str"),
        "E", np.array([[0, 0, 0], [0, 0, "e"], [0, "e", 0]], dtype = "str"),
        "E", np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype = "str")
    ],

    "-42m": [
        "A1", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "B1", np.array([["c", 0, 0], [0, "-c", 0], [0, 0, 0]], dtype = "str"),
        "B2", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str"),
        "E", np.array([[0, 0, 0], [0, 0, "e"], [0, "e", 0]], dtype = "str"),
        "E", np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype = "str")
    ],

    "4/mmm": [
        "A1g", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "B1g", np.array([["c", 0, 0], [0, "-c", 0], [0, 0, 0]], dtype = "str"),
        "B2g", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str"),
        "Eg", np.array([[0, 0, 0], [0, 0, "e"], [0, "e", 0]], dtype = "str"),
        "Eg", np.array([[0, 0, "-e"], [0, 0, 0], ["-e", 0, 0]], dtype = "str")
    ],

    "3": [
        "A(z)", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype= "str"),
        "1E", np.array([["c", "d", "e"], ["d", "-c", "f"], ["e", "f", 0]], dtype= "str"),
        "2E", np.array([["d", "-c", "-f"], ["-c", "-d", "e"], ["e", "-f", 0]], dtype= "str")
    ],

    "-3": [
        "Ag", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype= "str"),
        "1E", np.array([["c", "d", "e"], ["d", "-c", "f"], ["e", "f", 0]], dtype= "str"),
        "2E", np.array([["d", "-c", "-f"], ["-c", "-d", "e"], ["e", "-f", 0]], dtype= "str")
    ],

    "32": [
        "A1", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype= "str"),
        "Ex", np.array([["c", 0, 0], [0, "-c", "d"], [0, "d", 0]], dtype= "str"),
        "Ey", np.array([[0, "-c", "-d"], ["-c", 0, 0], ["-d", 0, 0]], dtype= "str")
    ],

    "3m": [
        "A1", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype= "str"),
        "E", np.array([[0, "c", "d"], ["c", 0, 0], ["d", 0, 0]], dtype= "str"),
        "E", np.array([["c", 0, 0], [0, "-c", "d"], [0, "d", 0]], dtype= "str"),
    ],

    "-3m": [
        "A1g", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype= "str"),
        "Eg,1", np.array([["c", 0, 0], [0, "-c", "d"], [0, "d", 0]], dtype= "str"),
        "Eg,2", np.array([[0, "-c", "-d"], ["-c", 0, 0], ["-d", 0, 0]], dtype= "str")    
    ],

    "6": [
        "A(z)", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "1E1", np.array([[0, 0, "c"], [0, 0, "d"], ["c", "d", 0]], dtype = "str"),
        "2E1", np.array([[0, 0, "-d"], [0, 0, "c"], ["-d", "c", 0]], dtype = "str"),
        "1E2", np.array([["e", "f", 0], ["f", "-e", 0], [0, 0, 0]], dtype = "str"),
        "2E2", np.array([["f", "-e", 0], ["-e", "-f", 0], [0, 0, 0]], dtype = "str")
    ],

    "-6": [
        "A'", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "1E''", np.array([[0, 0, "c"], [0, 0, "d"], ["c", "d", 0]], dtype = "str"),
        "2E''", np.array([[0, 0, "-d"], [0, 0, "c"], ["-d", "c", 0]], dtype = "str"),
        "1E'x", np.array([["e", "f", 0], ["f", "-e", 0], [0, 0, 0]], dtype = "str"),
        "2E'y", np.array([["f", "-e", 0], ["-e", "-f", 0], [0, 0, 0]], dtype = "str")
    ],

    "6/m": [
        "A'", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "1E1g", np.array([[0, 0, "c"], [0, 0, "d"], ["c", "d", 0]], dtype = "str"),
        "2E1g", np.array([[0, 0, "-d"], [0, 0, "c"], ["-d", "c", 0]], dtype = "str"),
        "1E2g", np.array([["e", "f", 0], ["f", "-e", 0], [0, 0, 0]], dtype = "str"),
        "2E2g", np.array([["f", "-e", 0], ["-e", "-f", 0], [0, 0, 0]], dtype = "str")
    ],

    "622": [
        "A1", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "E1", np.array([[0, 0, 0], [0, 0, "c"], [0, "c", 0]], dtype = "str"),
        "E1", np.array([[0, 0, "-c"], [0, 0, 0], ["-c", 0, 0]], dtype = "str"),
        "E2", np.array([["d", 0, 0], [0, "-d", 0], [0, 0, 0]], dtype = "str"),
        "E2", np.array([[0, "-d", 0], ["-d", 0, 0], [0, 0, 0]], dtype = "str")
    ],

    "6mm": [
        "A1", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "E1", np.array([[0, 0, "c"], [0, 0, 0], ["c", 0, 0]], dtype = "str"),
        "E1", np.array([[0, 0, 0], [0, 0, "c"], [0, "c", 0]], dtype = "str"),
        "E2", np.array([["d", 0, 0], [0, "-d", 0], [0, 0, 0]], dtype = "str"),
        "E2", np.array([[0, "-d", 0], ["-d", 0, 0], [0, 0, 0]], dtype = "str")
    ],

    "-62m": [
        "A'1", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "E''", np.array([[0, 0, 0], [0, 0, "c"], [0, "c", 0]], dtype = "str"),
        "E''", np.array([[0, 0, "-c"], [0, 0, 0], ["-c", 0, 0]], dtype = "str"),
        "E'", np.array([["d", 0, 0], [0, "-d", 0], [0, 0, 0]], dtype = "str"),
        "E'", np.array([[0, "-d", 0], ["-d", 0, 0], [0, 0, 0]], dtype = "str")
    ],

    "6/mmm": [
        "A1g", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype = "str"),
        "E1g", np.array([[0, 0, 0], [0, 0, "c"], [0, "c", 0]], dtype = "str"),
        "E1g", np.array([[0, 0, "-c"], [0, 0, 0], ["-c", 0, 0]], dtype = "str"),
        "E2g", np.array([["d", 0, 0], [0, "-d", 0], [0, 0, 0]], dtype = "str"),
        "E2g", np.array([[0, "-d", 0], ["-d", 0, 0], [0, 0, 0]], dtype = "str")
    ],

    "23": [
        "A", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
        "1E", np.array([["b+\u221A3c", 0, 0], [0, "b-\u221A3c", 0], [0, 0, "-2b"]], dtype = "str"),
        "2E", np.array([["c-\u221A3b", 0, 0], [0, "c+\u221A3b", 0], [0, 0, 0]], dtype = "str"),
        "T", np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype = "str"),
        "T", np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype = "str"),
        "T", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str")
    ],

    "m-3": [
        "Ag", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
        "1Eg", np.array([["b+\u221A3c", 0, 0], [0, "b-\u221A3c", 0], [0, 0, "-2b"]], dtype = "str"),
        "2Eg", np.array([["c-\u221A3b", 0, 0], [0, "c+\u221A3b", 0], [0, 0, 0]], dtype = "str"),
        "Tg", np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype = "str"),
        "Tg", np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype = "str"),
        "Tg", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str")
    ],

    "432": [
        "A1", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
        "E", np.array([["b", 0, 0], [0, "b", 0], [0, 0, "-2b"]], dtype = "str"),
        "E", np.array([["-\u221A3b", 0, 0], [0, "\u221A3b", 0], [0, 0, 0]], dtype = "str"),
        "T2", np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype = "str"),
        "T2", np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype = "str"),
        "T2", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str")
    ],

    "-43m": [
        "A1", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
        "E", np.array([["b", 0, 0], [0, "b", 0], [0, 0, "-2b"]], dtype = "str"),
        "E", np.array([["-\u221A3b", 0, 0], [0, "\u221A3b", 0], [0, 0, 0]], dtype = "str"),
        "T2", np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype = "str"),
        "T2", np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype = "str"),
        "T2", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str")
    ],

    "m-3m": [
        "A1g", np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),
        "Eg", np.array([["b", 0, 0], [0, "b", 0], [0, 0, "-2b"]], dtype = "str"),
        "Eg", np.array([["-\u221A3b", 0, 0], [0, "\u221A3b", 0], [0, 0, 0]], dtype = "str"),
        "T2g", np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype = "str"),
        "T2g", np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype = "str"),
        "T2g", np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str")
    ]
}


def RamanSelectionRules(pointgroup, RTs):
    if pointgroup == "1":
        backscattering = ["A",
                          "A",
                          "A",
                          "A",
                          "A",
                          "A",
                          "A",
                          "A",
                          "A"]
        backComponents = [RTs[1][bdDir[0]],\
                          RTs[1][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]],\
                          RTs[1][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]],\
                          RTs[1][bdDir[7]],\
                          RTs[1][bdDir[8]]]
        rightscattering = ["A",
                           "A",
                           "A",
                           "A",
                           "A",
                           "A",
                           "A",
                           "A",
                           "A", 
                           "A", 
                           "A",
                           "A"]
        rightComponents = [RTs[1][rDir[0]],\
                           RTs[1][rDir[1]],\
                           RTs[1][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[1][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[1][rDir[6]],\
                           RTs[1][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[1][rDir[9]],\
                           RTs[1][rDir[10]],\
                           RTs[1][rDir[11]]]
    
    elif pointgroup == "-1":
        backscattering = ["Ag",
                          "Ag",
                          "Ag",
                          "Ag",
                          "Ag",
                          "Ag",
                          "Ag",
                          "Ag",
                          "Ag"]
        backComponents = [RTs[1][bdDir[0]],\
                          RTs[1][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]],\
                          RTs[1][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]],\
                          RTs[1][bdDir[7]],\
                          RTs[1][bdDir[8]]]
        rightscattering = ["Ag",
                           "Ag",
                           "Ag",
                           "Ag",
                           "Ag",
                           "Ag",
                           "Ag",
                           "Ag",
                           "Ag",
                           "Ag",
                           "Ag",
                           "Ag"]
        rightComponents = [RTs[1][rDir[0]],\
                           RTs[1][rDir[1]],\
                           RTs[1][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[1][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[1][rDir[6]],\
                           RTs[1][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[1][rDir[9]],\
                           RTs[1][rDir[10]],\
                           RTs[1][rDir[11]]]
    
    elif pointgroup == "2":
        backscattering = ["A(TO)",   # x(yy)x
                          "B(LO+TO)",# x(yz)x
                          "A(TO)",   # x(zz)x
                          "A(TO)",   # y(xx)y
                          "B(LO+TO)",# y(xz)y
                          "A(TO)",   # y(zz)y
                          "A(LO)",   # z(xx)z
                          "A(LO)",   # z(xy)z
                          "A(LO)"]   # z(yy)z
        backComponents = [RTs[1][bdDir[0]],\
                          RTs[3][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]],\
                          RTs[3][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]],\
                          RTs[1][bdDir[7]],\
                          RTs[1][bdDir[8]]]
        rightscattering = ["A(TO)",    # x(yx)y
                           "B(LO)",    # x(yz)y
                           "B(LO)",    # x(zx)y
                           "A(TO)",    # x(zz)y
                           "A(LO+TO)", # x(yx)z
                           "A(LO+TO)", # x(yy)z
                           "B(LO+TO)", # x(zx)z
                           "B(LO+TO)", # x(zy)z
                           "A(LO+TO)", # y(xx)z
                           "A(LO+TO)", # y(xy)z
                           "B(LO+TO)", # y(zx)z
                           "B(LO+TO)"] # y(zy)z
        rightComponents = [RTs[1][rDir[0]],\
                           RTs[3][rDir[1]],\
                           RTs[3][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[1][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[3][rDir[6]],\
                           RTs[3][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[1][rDir[9]],\
                           RTs[3][rDir[10]],\
                           RTs[3][rDir[11]]]

    elif pointgroup == "m":
        backscattering = ["A'(LO+TO)",   # x(yy)x
                          "A''(TO)",     # x(yz)x
                          "A'(LO+TO)",   # x(zz)x
                          "A'(LO+TO)",   # y(xx)y
                          "A''(TO)",     # y(xz)y
                          "A'(LO+TO)",   # y(zz)y
                          "A'(TO)",      # z(xx)z
                          "A'(TO)",      # z(xy)z
                          "A'(TO)"]      # z(yy)z
        backComponents = [RTs[1][bdDir[0]],\
                          RTs[3][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]],\
                          RTs[3][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]],\
                          RTs[1][bdDir[7]],\
                          RTs[1][bdDir[8]]]
        rightscattering = ["A'(LO)",     # x(yx)y
                           "A''(TO)",    # x(yz)y
                           "A''(TO)",    # x(zx)y
                           "A'(LO)",     # x(zz)y
                           "A'(LO+TO)",  # x(yx)z
                           "A'(LO+TO)",  # x(yy)z
                           "A''(LO+TO)", # x(zx)z
                           "A''(LO+TO)", # x(zy)z
                           "A'(LO+TO)",  # y(xx)z
                           "A'(LO+TO)",  # y(xy)z
                           "A''(LO+TO)", # y(zx)z
                           "A''(LO+TO)"] # y(zy)z
        rightComponents = [RTs[1][rDir[0]],\
                           RTs[3][rDir[1]],\
                           RTs[3][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[1][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[3][rDir[6]],\
                           RTs[3][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[1][rDir[9]],\
                           RTs[3][rDir[10]],\
                           RTs[3][rDir[11]]]
        
    elif pointgroup == "2/m":
        backscattering = ["Ag",   # x(yy)x
                          "Bg",   # x(yz)x
                          "Ag",   # x(zz)x
                          "Ag",   # y(xx)y
                          "Bg",   # y(xz)y
                          "Ag",   # y(zz)y
                          "Ag",   # z(xx)z
                          "Ag",   # z(xy)z
                          "Ag"]   # z(yy)z
        backComponents = [RTs[1][bdDir[0]],\
                          RTs[3][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]],\
                          RTs[3][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]],\
                          RTs[1][bdDir[7]],\
                          RTs[1][bdDir[8]]]
        rightscattering = ["Ag", # x(yx)y
                           "Bg", # x(yz)y
                           "Bg", # x(zx)y
                           "Ag", # x(zz)y
                           "Ag", # x(yx)z
                           "Ag", # x(yy)z
                           "Bg", # x(zx)z
                           "Bg", # x(zy)z
                           "Ag", # y(xx)z
                           "Ag", # y(xy)z
                           "Bg", # y(zx)z
                           "Bg"] # y(zy)z
        rightComponents = [RTs[1][rDir[0]],\
                           RTs[3][rDir[1]],\
                           RTs[3][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[1][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[3][rDir[6]],\
                           RTs[3][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[1][rDir[9]],\
                           RTs[3][rDir[10]],\
                           RTs[3][rDir[11]]]
        
    elif pointgroup == "222":
        backscattering = ["A",       # x(yy)x
                          "B3(LO)",  # x(yz)x
                          "A",       # x(zz)x
                          "A",       # y(xx)y
                          "B2(LO)",  # y(xz)y
                          "A",       # y(zz)y
                          "A",       # z(xx)z
                          "B1(LO)",  # z(xy)z
                          "A"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]],\
                          RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]],\
                          RTs[3][bdDir[7]],\
                          RTs[1][bdDir[8]]]
        rightscattering = ["B1(TO)",    # x(yx)y
                           "B3(LO+TO)", # x(yz)y
                           "B2(LO+TO)", # x(zx)y
                           "A",         # x(zz)y
                           "B1(LO+TO)", # x(yx)z
                           "A",         # x(yy)z
                           "B2(TO)",    # x(zx)z
                           "B3(LO+TO)", # x(zy)z
                           "A",         # y(xx)z
                           "B1(LO+TO)", # y(xy)z
                           "B2(LO+TO)", # y(zx)z
                           "B3(TO)"]    # y(zy)z
        rightComponents = [RTs[3][rDir[0]],\
                           RTs[7][rDir[1]],\
                           RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[5][rDir[6]],\
                           RTs[7][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[3][rDir[9]],\
                           RTs[5][rDir[10]],\
                           RTs[7][rDir[11]]]
        
    elif pointgroup == "mm2":
        backscattering = ["A1(TO)", # x(yy)x
                          "B2(TO)", # x(yz)x
                          "A1(TO)", # x(zz)x
                          "A1(TO)", # y(xx)y
                          "B1(TO)", # y(xz)y
                          "A1(TO)", # y(zz)y
                          "A1(LO)", # z(xx)z
                          "A2",     # z(xy)z
                          "A1(LO)"] # z(yy)z
        backComponents = [RTs[1][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]],\
                          RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]],\
                          RTs[3][bdDir[7]],\
                          RTs[1][bdDir[8]]]
        rightscattering = ["A2",        # x(yx)y
                           "B2(LO+TO)", # x(yz)y
                           "B1(LO+TO)", # x(zx)y
                           "A1(TO)",    # x(zz)y
                           "A2",        # x(yx)z
                           "A1(LO+TO)", # x(yy)z
                           "B1(LO+TO)", # x(zx)z
                           "B2(TO)",    # x(zy)z
                           "A1(LO+TO)", # y(xx)z
                           "A2",        # y(xy)z
                           "B1(TO)",    # y(zx)z
                           "B2(LO+TO)"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]],\
                           RTs[7][rDir[1]],\
                           RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[5][rDir[6]],\
                           RTs[7][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[3][rDir[9]],\
                           RTs[5][rDir[10]],\
                           RTs[7][rDir[11]]]
        
    elif pointgroup == "mmm":
        backscattering = ["Ag",       # x(yy)x
                          "B3g",      # x(yz)x
                          "Ag",       # x(zz)x
                          "Ag",       # y(xx)y
                          "B2g",      # y(xz)y
                          "Ag",       # y(zz)y
                          "Ag",       # z(xx)z
                          "B1g",      # z(xy)z
                          "Ag"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]],\
                          RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]],\
                          RTs[3][bdDir[7]],\
                          RTs[1][bdDir[8]]]
        rightscattering = ["B1g", # x(yx)y
                           "B3g", # x(yz)y
                           "B2g", # x(zx)y
                           "Ag",  # x(zz)y
                           "B1g", # x(yx)z
                           "Ag",  # x(yy)z
                           "B2g", # x(zx)z
                           "B3g", # x(zy)z
                           "Ag",  # y(xx)z
                           "B1g", # y(xy)z
                           "B2g", # y(zx)z
                           "B3g"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]],\
                           RTs[7][rDir[1]],\
                           RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[5][rDir[6]],\
                           RTs[7][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[3][rDir[9]],\
                           RTs[5][rDir[10]],\
                           RTs[7][rDir[11]]]
        
    elif pointgroup == "3m":
        backscattering = ["A1(TO) + E(TO)",
                          "E(TO)",
                          "A1(TO)",
                          "A1(TO) + E(LO)",
                          "E(TO)", "A1(TO)",
                          "A1(LO) + E(TO)",
                          "E(TO)",
                          "A1(LO) + E(TO)"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[3][bdDir[7]]+RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["E(LO+TO)",
                           "E(LO+TO)",
                           "E(LO+TO)",
                           "A1(TO)",
                           "E(LO+TO)",
                           "A1(LO+TO)",
                           "E(LO+TO)",
                           "E(TO)",
                           "A1(LO+TO) + E(LO+TO)",
                           "E(TO)",
                           "E(TO)",
                           "E(LO+TO)"]
        rightComponents = [RTs[3][rDir[0]]+RTs[5][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]]+RTs[5][rDir[4]],\
                           RTs[1][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]]+RTs[5][rDir[8]],\
                           RTs[3][rDir[9]]+RTs[5][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
        
    elif pointgroup == "m-3m":
        backscattering = ["A1g + Eg",
                          "T2g",
                          "A1g + Eg",
                          "A1g + Eg",
                          "T2g",
                          "A1g + Eg",
                          "A1g + Eg",
                          "T2g",
                          "A1g + Eg"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[7][bdDir[1]]+RTs[9][bdDir[1]]+RTs[11][bdDir[1]],\
                          RTs[1][bdDir[2]]+RTs[3][bdDir[2]]+RTs[5][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[7][bdDir[4]]+RTs[9][bdDir[4]]+RTs[11][bdDir[4]],\
                          RTs[1][bdDir[5]]+RTs[3][bdDir[5]]+RTs[5][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]]+RTs[11][bdDir[7]],
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["T2g",
                           "T2g",
                           "T2g",
                           "A1g + Eg",
                           "T2g",
                           "A1g + Eg",
                           "T2g",
                           "T2g",
                           "A1g + Eg",
                           "T2g",
                           "T2g",
                           "T2g"]
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]]+RTs[11][rDir[0]],\
                           RTs[7][rDir[1]]+RTs[9][rDir[1]]+RTs[11][rDir[1]],\
                           RTs[7][rDir[2]]+RTs[9][rDir[2]]+RTs[11][rDir[2]],\
                           RTs[1][rDir[3]]+RTs[3][rDir[3]]+RTs[5][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]]+RTs[11][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]]+RTs[5][rDir[5]],\
                           RTs[7][rDir[6]]+RTs[9][rDir[6]]+RTs[11][rDir[6]],\
                           RTs[7][rDir[7]]+RTs[9][rDir[7]]+RTs[11][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]]+RTs[5][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]]+RTs[11][rDir[9]],\
                           RTs[7][rDir[10]]+RTs[9][rDir[10]]+RTs[11][rDir[10]],\
                           RTs[7][rDir[11]]+RTs[9][rDir[11]]+RTs[11][rDir[11]]]
    
    # just in case...
    else:
        print("[RamanLib]: no valid point group found, exiting...")
        sys.exit(1)
    #

    return backscattering, backComponents, rightscattering, rightComponents
#

IRSelectionRules = {
    "1": ["A",
          "A",
          "A"
    ],

    "-1":
        ["Au",
         "Au",
         "Au"
    ],

    "2":
        ["B",
         "B",
         "A"
    ],

    "m":
        ["A''",
         "A''",
         "A'"
    ],

    "2/m":
        ["Bu",
         "Bu",
         "Au"
    ],

    "222":
        ["B3",
         "B2",
         "B1"
    ],

    "mm2":
        ["B1",
         "B2",
         "A1"
    ],
    
    "mmm":
        ["B3u",
         "B2u",
         "B1u"
    ],

    "4":
        ["E",
         "E",
         "A2"
    ],

    "-4":
        ["E",
         "E",
         "A"
    ],

    "4/m":
        ["Eu",
         "Eu",
         "Au"
    ],

    "422":
        ["E",
         "E",
         "A2"
    ],

    "4mm":
        ["E",
         "E",
         "A1"
    ],

    "-42":
        ["E",
         "E",
         "B2"
    ],

    "4/mmm":
        ["Eu",
         "Eu",
         "A2u"
    ],

    "3":
        ["E",
         "E",
         "A2"
    ],

    "-3":
        ["Eu",
         "Eu",
         "Au"
    ],

    "32":
        ["E",
         "E",
         "A2"
    ],

    "3m":
        ["E",
         "E",
         "A1"
    ],

    "-3m":
        ["Eu",
         "Eu",
         "A2u"
    ],

    "6":
        ["E1",
         "E1",
         "A2"
    ],

    "-6":
        ["E1",
         "E1",
         "A"   
    ],

    "6/m":
        ["E1u",
         "E1u",
         "Au"
    ],

    "622":
        ["E1",
         "E1",
         "A2"
    ],

    "6mm":
        ["E1",
         "E1",
         "A1"
    ],

    "-62":
        ["E1",
         "E1",
         "B2"
    ],

    "6/mm":
        ["E1u",
         "E1u",
         "A2u"
    ],

    "23":
        ["T",
         "T",
         "T"
    ],

    "m-3":
        ["Tu",
         "Tu",
         "Tu"
    ],

    "432":
        ["T1",
         "T1",
         "T1"
    ],

    "-43m":
        ["T2",
         "T2",
         "T2"
    ],

    "m-3m":
        ["T1u",
         "T1u",
         "T1u"
    ]
}

def formatString(Component):
    newstring = ""
    stop = False
    dontadd = True
    for char in Component:
        if stop == True:
            if newstring == "":
                newstring += char
            else:
                newstring += "+"+char
            stop = False
            dontadd = True
        
        elif char == "\u221A":
            stop = True

        elif char.isdigit() and char != "0" and dontadd == False:
            newstring += "+"+str(int(char)**2)
            dontadd = True
        
        elif char.isdigit() and char != "0" and dontadd == True:
            newstring += str(int(char)**2)
            dontadd = True

        elif char != "0" and char != "-" and dontadd == False:
            newstring += "+"+char
            dontadd = False
            #
        elif char != "0" and char != "-" and dontadd == True:
            newstring += char
            dontadd = False
        #
    #

    # add up equal components
    printstring = ""
    terms = newstring.split("+")
    if terms != [""]:
        for j in range(len(terms)):
            if j < len(terms)-1 and terms[j][-1] in [x[-1] for x in terms[j+1:]]:
                k = j + 1 + [x[-1] for x in terms[j+1:]].index(terms[j][-1])
                if terms[j][:-1].isdigit():
                    factor1 = int(terms[j][:-1])
                else:
                    factor1 = 1
                if terms[k][:-1].isdigit():
                    factor2 = int(terms[k][:-1])
                else:
                    factor2 = 1
                terms[k] = str(factor1+factor2)+terms[k][-1]
            else:
                if printstring == "":
                    printstring += terms[j]+"^2"
                else:
                    printstring += " + " + terms[j]+"^2"
                #
            #
        #
    #
    return printstring
#

def flatten(t):
    a = []
    for sublist in t:
        if isinstance(sublist, str):
            a.append(sublist)
        else:
            for item in sublist:
                a.append(item)
            #
        #
    #
    return a
#

def Lorentz(hw, ab, gam=0.001):
    fmax = max(hw)
    erange = np.arange(0, 1.1*fmax, gam/10)
    spectrum = 0.0 * erange
    for i in range(len(hw)):
        spectrum +=  ab[i] * gam  / ( (hw[i]-erange)**2 + gam**2 )
    #
    return erange, spectrum
#

def placzeck(Intensity, col):
    # get Placzeck-invariants
    G0 = np.abs(Intensity[0][col-1] + Intensity[1][col-1] + Intensity[2][col-1])**2/3.0
    G1 = 0
    G2 = (np.abs(Intensity[0][col-1] - Intensity[1][col-1])**2 \
          + np.abs(Intensity[0][col-1] - Intensity[2][col-1])**2 \
          + np.abs(Intensity[1][col-1] - Intensity[2][col-1])**2)/3.0 \
          + 2*(np.abs(Intensity[3][col-1])**2 + np.abs(Intensity[4][col-1])**2 + np.abs(Intensity[5][col-1])**2)
    avg = np.sqrt(10*G0 + 5*G1 + 7*G2) # parallel and perpendicular components added together
    return avg
#

def classifyRotations(rotations):
    classes = {}
    for i, R in enumerate(rotations):
        det = round(np.linalg.det(R))
        trace = np.trace(R)
        key = (det, trace)
        classes.setdefault(key, []).append(i)
    return list(classes.values())
#

def matchLabels(class_characters, char_table):
    labels = []
    table_labels = list(char_table.keys())
    table_rows = np.array(list(char_table.values()), dtype=float)

    for chi in class_characters:
        dists = np.sum((table_rows - chi)**2, axis=1)
        best = np.argmin(dists)
        labels.append(table_labels[best])
    return labels
#

def getIrrepsSymbols(path, basis, coord, elements, pointgroup):
    # set up a phonopy structure and get irreps
    # works for phonopy 2.32
    import os
    pwd = os.getcwd()
    os.chdir(path)
    cell = PhonopyAtoms( symbols=elements, cell=basis, scaled_positions=coord )
    phonopy_instance = phonopy.load(unitcell=cell, supercell_matrix=np.eye(3), primitive_matrix="auto", force_constants_filename="FORCE_CONSTANTS")
    os.chdir(pwd)

    ir = IrReps(phonopy_instance.dynamical_matrix, q=[0, 0, 0])
    ir.run()

    characters = ir.get_characters() # shape: (n_irreps, n_sym_ops)
    rotations = ir.get_rotations() # shape: (n_sym_ops, 3, 3)
    classes = classifyRotations(rotations)

    # class_characters[i_irrep][i_class]
    class_characters = []
    for chi in characters:
        class_row = [np.mean(chi[idxs]) for idxs in classes]
        class_characters.append(class_row)
    #
    class_characters = np.array(class_characters)
    
    labels = matchLabels(class_characters, CHAR_TABLES[HM_TO_SCHOENFLIES[pointgroup]])

    return labels
#

def getAcoustics(eigvecs, eigvals, masses): 
    # get the candidates for possible acoustic modes
    acoustic = []
    for j in range(len(eigvals)):
        if np.abs(eigvals[j]) < 10:
            acoustic.append(j)
        #
    #   
    masses = np.array(masses)
    sqrt_m = np.sqrt(masses)
    
    indicators = np.empty(len(eigvecs))
    for j in range(len(eigvecs)):
        mode = np.array(eigvecs[j])
        S = np.sum(sqrt_m[:, None] * mode, axis=0) 
        indicators[j] = np.linalg.norm(S)
    #
    if np.array_equal( np.sort(indicators.argsort()[-3:]), np.sort(acoustic) ) or \
       np.array_equal( np.sort(indicators.argsort()[-2:]), np.sort(acoustic) ) or \
       np.array_equal( np.sort(indicators.argsort()[-1:]), np.sort(acoustic) ):
        return [x+1 for x in acoustic]
    else:
        print("[removeAcoustics]: Could not determine acoustic modes, continuing...")
        return []
    #
#

def getDegenerates(eigvals, labels, prec=1e0):
    degenerates = []
    for j in range(len(eigvals)):
        for k in range(j+1, len(eigvals)):
            tmp = []
            if np.abs(eigvals[j] - eigvals[k]) < prec and labels[j] == labels[k]:
                if j not in tmp:
                    tmp.append(j+1)
                #
                tmp.append(k+1)
            #
            if tmp != []:
                degenerates.append(tmp)
            #
        #
    #
    return degenerates
#

def getSilent(modelist, labels, pointgroup):
    RamanTensors = RamanTensorComponents[pointgroup]
    silent = []
    for mode in modelist:
        if labels[mode-1] not in RamanTensors[::2]:
            silent.append(mode)
        #
    #
    return silent

def removeModes(eigvecs, eigvals, masses, modelist, basis, coord, elements, pointgroup, prec=1e0):
    modelist_new = []
    labels = getIrrepsSymbols(basis, coord, elements, pointgroup)
    acoustics = getAcoustics(eigvecs, eigvals, masses)
    degenerates = getDegenerates(eigvals, labels, prec)
    silent = getSilent(modelist, labels, pointgroup)
    check_acoustic = False
    check_imag = False
    check_degenerates = False
    check_silent = False
    for mode in modelist:
        if mode in acoustics:
            check_acoustic = True
        elif eigvals[mode-1] < 0:
            check_imag = True
        elif mode in degenerates:
            check_degenerates = True
        elif mode in silent:
            check_silent = True
        else:
            modelist_new.append(mode)
        #
    #

    if check_acoustic == True:
        print("[removeModes]: Ignoring acoustic modes")
    if check_imag == True:
        print("[removeModes]: Ignoring modes with imaginary frequency")
    if check_degenerates == True:
        print("[removeModes]: Ignoring degenerate modes")
    if check_silent == True:
        print("[removeModes]: Ignoring Raman silent modes")
    #

    return modelist_new
#

def getBorn(path, program, nat):
    # get BORN charges, in |e|
    if program == "VASP":
        from parserVASP import getBornVASP
        born = getBornVASP(path+"OUTCAR", nat)
    elif program == "QE":
        from parserQE import getBornQE
        born = getBornQE(path+"ph.out", nat)
    else:
        print("[getBorn]: Format not implemented, exiting..")
        born = []
    #
    return born
#

def getEpsInf(path, program):
    # get ion-clamped static dielectric function
    if program == "phonopy":
        try:
            phonopy_fh = open(path+"BORN", "r")
        except IOError:
            print("[getEpsInf]: ERROR Couldn't open BORN")
        lines = [l.strip() for l in phonopy_fh.readlines()] # Read the whole file removing tailoring spaces
        phonopy_fh.close()

        data = lines[1].split()
        EpsInf = np.array([[float(data[0]), float(data[1]), float(data[2])], [float(data[3]), float(data[4]), float(data[5])], [float(data[6]), float(data[7]), float(data[8])]])
    elif program == "VASP":
        from parserVASP import getEpsInfVASP
        born = getEpsInfVASP(path+"OUTCAR")
    elif program == "QE":
        from parserQE import getEpsInfQE
        born = getEpsInfQE(path+"ph.out")
    else:
        print("[getEpsInf]: Format not implemented\n")
        EpsInf = []
    #
    return EpsInf
#