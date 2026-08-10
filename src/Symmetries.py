#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import os, re
import numpy as np
import heapq
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.irreps import IrReps

#####################
# physical constants
#####################
e_charge = 1.602176634e-19   # C
amu      = 1.66053906660e-27 # kg
eps0     = 8.8541878128e-12  # F/m
c_cm     = 2.99792458e10     # cm/s
h        = 6.62606957e-34    # Js
kb       = 1.3806488e-23     # J/K
ev2rcm   = 8065.5401

#####################
# elements
#####################

periodTable = {'': 0, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
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
periodTableMasses = {'': 0, 'H':  1.00797, 'He':  4.00260, 'Li':  6.941,
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

##################
# polarization and propagation directions of photons
##################

backDirs = ["x(yy)x\u0305", "x(yz)x\u0305", "x(zz)x\u0305", "y(xx)y\u0305", "y(xz)y\u0305", "y(zz)y\u0305", "z(xx)z\u0305", "z(xy)z\u0305", "z(yy)z\u0305"]
bdDir = {0: (1,1), 1: (1,2), 2: (2,2), 3: (0,0), 4: (0,2), 5: (2,2), 6: (0,0), 7: (0,1), 8: (1,1)}
rightDirs = ["x(yx)y", "x(yz)y", "x(zx)y", "x(zz)y", "x(yx)z", "x(yy)z", "x(zx)z", "x(zy)z", "y(xx)z", "y(xy)z", "y(zx)z", "y(zy)z"]
rDir = {0: (0,1), 1: (1,2), 2: (0,2), 3: (2,2), 4: (0,1), 5: (1,1), 6: (0,2), 7: (1,2), 8: (0,0), 9:(0,1), 10: (0,2), 11: (1,2)}
IRDirs = ["E || x", "E || y", "E || z"]
portoq = {(0, 0, 1) : "zz", (0, 1, 0) : "yy", (1, 0, 0) : "xx", (1, 1, 0) : "xy", (1, 0, 1): "xz", (0, 1, 1) : "yz" }

###################
# pointgroup information
###################

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
        "classes": ["E"],
        "irreps": {
        "A": [1],}
    },

    "Ci": {
        "classes": ["E", "i"],
        "irreps": {
        "Ag": [1, 1],
        "Au": [1, -1],}
    },

    # 2. Monoclinic
    "C2": {
        "classes": ["E", "C2"],
        "irreps": {
        "A": [1, 1],
        "B": [1, -1],}
    },

    "Cs": {
        "classes": ["E", "σ"],
        "irreps": {
        "A'": [1, 1],
        "A''": [1, -1],}
    },

    "C2h": {
        "classes": ["E", "C2", "i", "σh"],
        "irreps": {
        "Ag": [1, 1, 1, 1],
        "Bg": [1, -1, 1, -1],
        "Au": [1, 1, -1, -1],
        "Bu": [1, -1, -1, 1],}
    },

    # 3. Orthorhombic
    "D2": {
        "classes": ["E", "C2x", "C2y", "C2z"],
        "irreps": {
        "A": [1, 1, 1, 1],
        "B1": [1, 1, -1, -1],
        "B2": [1, -1, 1, -1],
        "B3": [1, -1, -1, 1],}
    },

    "C2v": {
        "classes": ["E", "C2", "σv", "σv'"],
        "irreps": {
        "A1": [1, 1, 1, 1],
        "A2": [1, 1, -1, -1],
        "B1": [1, -1, 1, -1],
        "B2": [1, -1, -1, 1],}
    },

    "D2h": {
        "classes": ["E", "C2x", "C2y", "C2z", "i", "σxy", "σxz", "σyz"],
        "irreps": {
        "Ag": [1, 1, 1, 1, 1, 1, 1, 1],
        "B1g": [1, 1, -1, -1, 1, 1, -1, -1],
        "B2g": [1, -1, 1, -1, 1, -1, 1, -1],
        "B3g": [1, -1, -1, 1, 1, -1, -1, 1],
        "Au": [1, 1, 1, 1, -1, -1, -1, -1],
        "B1u": [1, 1, -1, -1, -1, -1, 1, 1],
        "B2u": [1, -1, 1, -1, -1, 1, -1, 1],
        "B3u": [1, -1, -1, 1, -1, 1, 1, -1],}
    },

    # 4. Tetragonal
    "C4": {
        "classes": ["E", "C4", "C2", "C4^3"],
        "irreps": {
        "A": [1, 1, 1, 1],
        "B": [1, -1, 1, -1],
        "E": [2, 0, -2, 0],}
    },

    "S4": {
        "classes": ["E", "S4", "C2", "S4^3"],
        "irreps": {
        "A": [1, 1, 1, 1],
        "B": [1, -1, 1, -1],
        "E": [2, 0, -2, 0],}
    },

    "C4h": {
        "classes": ["E", "C4", "C2", "C4^3", "i", "S4", "σh", "S4^3"],
        "irreps": {
        "Ag": [1, 1, 1, 1, 1, 1, 1, 1],
        "Bg": [1, -1, 1, -1, 1, -1, 1, -1],
        "Eg": [2, 0, -2, 0, 2, 0, -2, 0],
        "Au": [1, 1, 1, 1, -1, -1, -1, -1],
        "Bu": [1, -1, 1, -1, -1, 1, -1, 1],
        "Eu": [2, 0, -2, 0, -2, 0, 2, 0],}
    },

    "C4v": {
        "classes": ["E", "2C4", "C2", "2σv", "2σd"],
        "irreps": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "B1": [1, -1, 1, 1, -1],
        "B2": [1, -1, 1, -1, 1],
        "E": [2, 0, -2, 0, 0],}
    },

    "D4": {
        "classes": ["E", "2C4", "C2", "2C2'", "2C2''"],
        "irreps": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "B1": [1, -1, 1, 1, -1],
        "B2": [1, -1, 1, -1, 1],
        "E": [2, 0, -2, 0, 0],}
    },

    "D2d": {
        "classes": ["E", "2S4", "C2", "2C2'", "2σd"],
        "irreps": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "B1": [1, -1, 1, 1, -1],
        "B2": [1, -1, 1, -1, 1],
        "E": [2, 0, -2, 0, 0],}
    },

    "D4h": {
        "classes": ["E", "2C4", "C2", "2C2'", "2C2''", "i", "2S4", "σh", "2σv", "2σd"],
        "irreps": {
        "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "A2g": [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
        "B1g": [1, -1, 1, 1, -1, 1, -1, 1, 1, -1],
        "B2g": [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
        "Eg": [2, 0, -2, 0, 0, 2, 0, -2, 0, 0],
        "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
        "A2u": [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
        "B1u": [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        "B2u": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        "Eu": [2, 0, -2, 0, 0, -2, 0, 2, 0, 0],}
    },

    # 5. Trigonal
    "C3": {
        "classes": ["E", "2C3"],
        "irreps": {
        "A": [1, 1, 1],
        "E": [2, -1, -1],}
    },

    "S6": {
        "classes": ["E", "C3", "C3^2", "i", "S6", "S6^5"],
        "irreps": {
        "Ag": [1, 1, 1, 1, 1, 1],
        "Au": [1, 1, 1, -1, -1, -1],
        "Eg": [2, -1, -1, 2, -1, -1],
        "Eu": [2, -1, -1, -2, 1, 1],}
    },

    "C3v": {
        "classes": ["E", "2C3", "3σv"],
        "irreps": {
        "A1": [1, 1, 1],
        "A2": [1, 1, -1],
        "E": [2, -1, 0],}
    },

    "D3": {
        "classes": ["E", "2C3", "3C2"],
        "irreps": {
        "A1": [1, 1, 1],
        "A2": [1, 1, -1],
        "E": [2, -1, 0],}
    },

    "D3d": {
        "classes": ["E", "2C3", "3C2", "i", "2S6", "3σd"],
        "irreps": {
        "A1g": [1, 1, 1, 1, 1, 1],
        "A2g": [1, 1, -1, 1, 1, -1],
        "Eg": [2, -1, 0, 2, -1, 0],
        "A1u": [1, 1, 1, -1, -1, -1],
        "A2u": [1, 1, -1, -1, -1, 1],
        "Eu": [2, -1, 0, -2, 1, 0],}
    },

    # 6. Hexagonal
    "C6": {
        "classes": ["E", "2C6", "2C3", "C2"],
        "irreps": {
        "A": [1, 1, 1, 1, 1, 1],
        "B": [1, -1, 1, -1, 1, -1],
        "E1": [2, 1, -1, -2, -1, 1],
        "E2": [2, -1, -1, 2, -1, -1],}
    },

    "C3h": {
        "classes": ["E", "C3", "C3^2", "σh", "S3", "S3^5"],
        "irreps": {
        "A'": [1, 1, 1, 1, 1, 1],
        "A''": [1, 1, 1, -1, -1, -1],
        "E'": [2, -1, -1, 2, -1, -1],
        "E''": [2, -1, -1, -2, 1, 1],}
    },

    "C6h": {
        "classes": ["E", "2C6", "2C3", "C2", "i", "2S3", "2S6", "σh"],
        "irreps": {
        "Ag": [1]*12,
        "Bg": [1, -1]*6,
        "E1g": [2, 1, -1, -2, -1, 1]*2,
        "E2g": [2, -1, -1, 2, -1, -1]*2,
        "Au": [1]*6 + [-1]*6,
        "Bu": [1, -1]*3 + [-1, 1]*3,
        "E1u": [2, 1, -1, -2, -1, 1] + [-2, -1, 1, 2, 1, -1],
        "E2u": [2, -1, -1, 2, -1, -1] + [-2, 1, 1, -2, 1, 1],}
    },

    "C6v": {
        "classes": ["E", "2C6", "2C3", "C2", "3σv", "3σd"],
        "irreps": {
        "A1": [1, 1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1, -1],
        "B1": [1, -1, 1, 1, -1, 1],
        "B2": [1, -1, 1, -1, 1, -1],
        "E1": [2, 1, -1, 0, -1, 1],
        "E2": [2, -1, -1, 0, -1, -1],}
    },

    "D6": {
        "classes": ["E", "2C6", "2C3", "C2", "3C2'", "3C2''"],
        "irreps": {
        "A1": [1, 1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1, -1],
        "B1": [1, -1, 1, 1, -1, 1],
        "B2": [1, -1, 1, -1, 1, -1],
        "E1": [2, 1, -1, 0, -1, 1],
        "E2": [2, -1, -1, 0, -1, -1],}
    },

    "D3h": {
        "classes": ["E", "2C3", "3C2'", "σh", "2S3", "3σv"],
        "irreps": {
        "A1'": [1, 1, 1, 1, 1, 1],
        "A2'": [1, 1, -1, 1, 1, -1],
        "E'": [2, -1, 0, 2, -1, 0],
        "A1''": [1, 1, 1, -1, -1, -1],
        "A2''": [1, 1, -1, -1, -1, 1],
        "E''": [2, -1, 0, -2, 1, 0],}
    },

    "D6h": {
        "classes": ["E", "2C6", "2C3", "C2", "3C2'", "3C2''",
                   "i", "2S3", "2S6", "σh", "3σv", "3σd"],
        "irreps": {
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
        "E2u": [2, -1, -1, 0, -1, -1] + [-2, 1, 1, 0, 1, 1],}
    },

    # 7. Cubic
    "T": {
        "classes": ["E", "8C3", "3C2"],
        "irreps": {
        "A": [1, 1, 1],
        "E": [2, -1, 2],
        "T": [3, 0, -1],}
    },

    "Th": {
        "classes": ["E", "8C3", "3C2", "i", "8S6", "3σh"],
        "irreps": {
        "Ag": [1, 1, 1, 1, 1, 1],
        "Eg": [2, -1, 2, 2, -1, 2],
        "Tg": [3, 0, -1, 3, 0, -1],
        "Au": [1, 1, 1, -1, -1, -1],
        "Eu": [2, -1, 2, -2, 1, -2],
        "Tu": [3, 0, -1, -3, 0, 1],}
    },

    "O": {
        "classes": ["E", "8C3", "6C4", "3C2", "6C2'"],
        "irreps": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "E": [2, -1, 2, 0, 0],
        "T1": [3, 0, -1, 1, -1],
        "T2": [3, 0, -1, -1, 1],}
    },

    "Td": {
        "classes": ["E", "8C3", "3C2", "6S4", "6σd"],
        "irreps": {
        "A1": [1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1],
        "E": [2, -1, 2, 0, 0],
        "T1": [3, 0, -1, 1, -1],
        "T2": [3, 0, -1, -1, 1],}
    },

    "Oh": {
        "classes": ["E", "8C3", "6C4", "3C2", "6C2'",
                  "i", "8S6", "6S4", "3σh", "6σd"],
        "irreps": {
        "A1g": [1]*10,
        "A2g": [1, 1, 1, -1, -1]*2,
        "Eg": [2, -1, 2, 0, 0]*2,
        "T1g": [3, 0, -1, 1, -1]*2,
        "T2g": [3, 0, -1, -1, 1]*2,
        "A1u": [1]*5 + [-1]*5,
        "A2u": [1, 1, 1, -1, -1] + [-1, -1, -1, 1, 1],
        "Eu": [2, -1, 2, 0, 0] + [-2, 1, -2, 0, 0],
        "T1u": [3, 0, -1, 1, -1] + [-3, 0, 1, -1, 1],
        "T2u": [3, 0, -1, -1, 1] + [-3, 0, 1, 1, -1],}
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
     "m-3m": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),

     # molecules
     "Cinfv": np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype = "str"),

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
    
    elif pointgroup == "4":
        backscattering = ["A(TO) + B",       # x(yy)x
                          "2E(TO)",      # x(yz)x
                          "A(TO)",       # x(zz)x
                          "A(TO) + B",       # y(xx)y
                          "1E(TO) + 2E(LO)",      # y(xz)y
                          "A(TO)",       # y(zz)y
                          "A(LO) + B",       # z(xx)z
                          "B",      # z(xy)z
                          "A(LO) + B"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]],\
                          RTs[5][bdDir[4]]+RTs[7][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]],\
                          RTs[3][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]]
        rightscattering = ["B", # x(yx)y
                           "E1(LO+TO) + E2(LO+TO)", # x(yz)y
                           "E1(LO+TO) + E2(LO+TO)", # x(zx)y
                           "A(TO)",  # x(zz)y
                           "B", # x(yx)z
                           "A(LO+TO) + B",  # x(yy)z
                           "E1(LO+TO) + E2(TO)", # x(zx)z
                           "E1(LO+TO) + E2(TO)", # x(zy)z
                           "A(TO)",  # y(xx)z
                           "B", # y(xy)z
                           "E1(TO) + E2(LO+TO)", # y(zx)z
                           "E1(TO) + E2(LO+TO)"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]],\
                           RTs[5][rDir[1]]+RTs[7][rDir[1]],\
                           RTs[5][rDir[2]]+RTs[7][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][bdDir[5]],\
                           RTs[5][rDir[6]]+RTs[7][rDir[6]],\
                           RTs[5][rDir[7]]+RTs[7][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[3][rDir[9]],\
                           RTs[5][rDir[10]]+RTs[7][rDir[10]],\
                           RTs[5][rDir[11]]+RTs[7][rDir[11]]]
    
    elif pointgroup == "-4":
        backscattering = ["A + B(TO)",       # x(yy)x
                          "E1(LO) + E2(TO)",      # x(yz)x
                          "A",       # x(zz)x
                          "A + B(TO)",       # y(xx)y
                          "E1(TO) + E2(LO)",      # y(xz)y
                          "A",       # y(zz)y
                          "A + B(LO)",       # z(xx)z
                          "B(LO)",      # z(xy)z
                          "A + B(LO)"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]],\
                          RTs[5][bdDir[4]]+RTs[7][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]],\
                          RTs[3][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]]
        rightscattering = ["B(TO)", # x(yx)y
                           "E1(LO+TO) + E2(LO+TO)", # x(yz)y
                           "E1(LO+TO) + E2(LO+TO)", # x(zx)y
                           "A",  # x(zz)y
                           "B(LO+TO)", # x(yx)z
                           "A + B(LO+TO)",  # x(yy)z
                           "E1(LO+TO) + E2(TO)", # x(zx)z
                           "E1(LO+TO) + E2(TO)", # x(zy)z
                           "A +  B(LO+TO)",  # y(xx)z
                           "B(LO+TO)", # y(xy)z
                           "E1(TO) + E2(LO+TO)", # y(zx)z
                           "E1(TO) + E2(LO+TO)"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]],\
                           RTs[5][rDir[1]]+RTs[7][rDir[1]],\
                           RTs[5][rDir[2]]+RTs[7][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][bdDir[5]],\
                           RTs[5][rDir[6]]+RTs[7][rDir[6]],\
                           RTs[5][rDir[7]]+RTs[7][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[3][rDir[9]],\
                           RTs[5][rDir[10]]+RTs[7][rDir[10]],\
                           RTs[5][rDir[11]]+RTs[7][rDir[11]]]
        
    elif pointgroup == "4/m":
        backscattering = ["Ag + Bg",       # x(yy)x
                          "1Eg + 2Eg",      # x(yz)x
                          "Ag",       # x(zz)x
                          "Ag + Bg",       # y(xx)y
                          "1Eg + 2Eg",      # y(xz)y
                          "Ag",       # y(zz)y
                          "Ag + Bg",       # z(xx)z
                          "Bg",      # z(xy)z
                          "Ag + Bg"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]],\
                          RTs[5][bdDir[1]]+RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]],\
                          RTs[5][bdDir[4]]+RTs[7][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]],\
                          RTs[3][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]]
        rightscattering = ["Bg", # x(yx)y
                           "1Eg + 2Eg", # x(yz)y
                           "1Eg + 2Eg", # x(zx)y
                           "Ag",  # x(zz)y
                           "Bg", # x(yx)z
                           "Ag + Bg",  # x(yy)z
                           "1Eg + 2Eg", # x(zx)z
                           "1Eg + 2Eg", # x(zy)z
                           "Ag",  # y(xx)z
                           "Bg", # y(xy)z
                           "1Eg + 2Eg", # y(zx)z
                           "1Eg + 2Eg"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]],\
                           RTs[5][rDir[1]]+RTs[7][rDir[1]],\
                           RTs[5][rDir[2]]+RTs[7][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][bdDir[5]],\
                           RTs[5][rDir[6]]+RTs[7][rDir[6]],\
                           RTs[5][rDir[7]]+RTs[7][rDir[7]],\
                           RTs[1][rDir[8]],\
                           RTs[3][rDir[9]],\
                           RTs[5][rDir[10]]+RTs[7][rDir[10]],\
                           RTs[5][rDir[11]]+RTs[7][rDir[11]]]
        
    elif pointgroup == "422":
        backscattering = ["A1 + B1",       # x(yy)x
                          "E(LO)",      # x(yz)x
                          "A1",       # x(zz)x
                          "A1 + B1",       # y(xx)y
                          "E(LO)",      # y(xz)y
                          "A1",       # y(zz)y
                          "A1 + B1",       # z(xx)z
                          "B2",      # z(xy)z
                          "A1 + B1"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]],\
                          RTs[7][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]],\
                          RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]]
        rightscattering = ["B2", # x(yx)y
                           "E(LO+TO)", # x(yz)y
                           "E(LO+TO)", # x(zx)y
                           "A1",  # x(zz)y
                           "B2", # x(yx)z
                           "A1 + B1",  # x(yy)z
                           "E(TO)", # x(zx)z
                           "E(LO+TO)", # x(zy)z
                           "A1 + B1",  # y(xx)z
                           "B2", # y(xy)z
                           "E(LO+TO)", # y(zx)z
                           "E(TO)"] # y(zy)z
        rightComponents = [RTs[5][rDir[0]],\
                           RTs[7][rDir[1]],\
                           RTs[7][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[5][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]],\
                           RTs[7][rDir[6]],\
                           RTs[7][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]],\
                           RTs[5][rDir[9]],\
                           RTs[7][rDir[10]],\
                           RTs[7][rDir[11]]]
        
    elif pointgroup == "4mm":
        backscattering = ["A(TO) + B1",       # x(yy)x
                          "E(TO)",      # x(yz)x
                          "A(TO)",       # x(zz)x
                          "A(TO) + B1",       # y(xx)y
                          "E(TO)",      # y(xz)y
                          "A(TO)",       # y(zz)y
                          "A(LO) + B1",       # z(xx)z
                          "B2",      # z(xy)z
                          "A(LO) + B1"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]],\
                          RTs[7][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]],\
                          RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]]
        rightscattering = ["B2", # x(yx)y
                           "E(LO+TO)", # x(yz)y
                           "E(LO+TO)", # x(zx)y
                           "A(TO)",  # x(zz)y
                           "B2", # x(yx)z
                           "A(LO+TO) + B1",  # x(yy)z
                           "E(TO)", # x(zx)z
                           "E(LO+TO)", # x(zy)z
                           "A(LO+TO) + B1",  # y(xx)z
                           "B2", # y(xy)z
                           "E(TO)", # y(zx)z
                           "E(LO+TO)"] # y(zy)z
        rightComponents = [RTs[5][rDir[0]],\
                           RTs[7][rDir[1]],\
                           RTs[7][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[5][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]],\
                           RTs[7][rDir[6]],\
                           RTs[7][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]],\
                           RTs[5][rDir[9]],\
                           RTs[7][rDir[10]],\
                           RTs[7][rDir[11]]]
        
    elif pointgroup == "-42m":
        backscattering = ["A1 + B1",       # x(yy)x
                          "E(LO)",      # x(yz)x
                          "A1",       # x(zz)x
                          "A1 + B1",       # y(xx)y
                          "E(LO)",      # y(xz)y
                          "A1",       # y(zz)y
                          "A1 + B1",       # z(xx)z
                          "B2(LO)",      # z(xy)z
                          "A1 + B1"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]],\
                          RTs[7][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]],\
                          RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]]
        rightscattering = ["B2(TO)", # x(yx)y
                           "E(LO+TO)", # x(yz)y
                           "E(LO+TO)", # x(zx)y
                           "A1",  # x(zz)y
                           "B2(LO+TO)", # x(yx)z
                           "A1 + B1",  # x(yy)z
                           "E(TO)", # x(zx)z
                           "E(LO+TO)", # x(zy)z
                           "A1 + B1",  # y(xx)z
                           "B2(LO+TO)", # y(xy)z
                           "E(LO+TO)", # y(zx)z
                           "E(TO)"] # y(zy)z
        rightComponents = [RTs[5][rDir[0]],\
                           RTs[7][rDir[1]],\
                           RTs[7][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[5][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]],\
                           RTs[7][rDir[6]],\
                           RTs[7][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]],\
                           RTs[5][rDir[9]],\
                           RTs[7][rDir[10]],\
                           RTs[7][rDir[11]]]
        
    elif pointgroup == "4/mmm":
        backscattering = ["A1g + B1g",       # x(yy)x
                          "Eg",      # x(yz)x
                          "A1g",       # x(zz)x
                          "A1g + B1g",       # y(xx)y
                          "Eg",      # y(xz)y
                          "A1g",       # y(zz)y
                          "A1g + B1g",       # z(xx)z
                          "B2g",      # z(xy)z
                          "A1g + B1g"]       # z(yy)z
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]],\
                          RTs[7][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]],\
                          RTs[7][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]],\
                          RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]]
        rightscattering = ["B2g", # x(yx)y
                           "Eg", # x(yz)y
                           "Eg", # x(zx)y
                           "A1g",  # x(zz)y
                           "B2g", # x(yx)z
                           "A1g + B1g",  # x(yy)z
                           "Eg", # x(zx)z
                           "Eg", # x(zy)z
                           "A1g + B1g",  # y(xx)z
                           "B2g", # y(xy)z
                           "Eg", # y(zx)z
                           "Eg"] # y(zy)z
        rightComponents = [RTs[5][rDir[0]],\
                           RTs[7][rDir[1]],\
                           RTs[7][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[5][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]],\
                           RTs[7][rDir[6]],\
                           RTs[7][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]],\
                           RTs[5][rDir[9]],\
                           RTs[7][rDir[10]],\
                           RTs[7][rDir[11]]]
    
    elif pointgroup == "3":
        backscattering = ["A(TO) + 1E(LO) + 2E(TO)",
                          "1E(LO) + 2E(TO)",
                          "A(TO)",
                          "A(TO) + 1E(TO) + 2E(LO)",
                          "1E(TO)+ 2E(LO)",
                          "A(TO)",
                          "A(LO) + 1E(TO) + 2E(TO)",
                          "1E(TO) + 2E(TO)",
                          "A(LO) + 1E(TO) + 2E(TO)"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[3][bdDir[7]]+RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["1E(LO+TO) + 2E(LO+TO)",  # x(yx)y
                           "1E(LO+TO) + 2E(LO+TO)", # x(yz)y
                           "1E(LO+TO) + 2E(LO+TO)",  # x(zx)y
                           "A(TO)",  # x(zz)y
                           "1E(LO+TO) + 2E(TO)", # x(yx)z
                           "1E(LO+TO) + 2E(TO)",  # x(yy)z
                           "1E(LO+TO) + 2E(TO)", # x(zx)z
                           "1E(LO+TO) + 2E(TO)", # x(zy)z
                           "1E(TO) + 2E(LO+TO)", # y(xx)z
                           "1E(TO) + 2E(LO+TO)",  # y(xy)z
                           "1E(TO) + 2E(LO+TO)", # y(zx)z
                           "1E(TO) + 2E(LO+TO)"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]]+RTs[5][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]]+RTs[5][rDir[4]],\
                           RTs[3][rDir[5]]+RTs[5][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[3][rDir[8]]+RTs[5][rDir[8]],\
                           RTs[3][rDir[9]]+RTs[5][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
        
    elif pointgroup == "-3":
        backscattering = ["Ag + 1Eg + 2Eg",
                          "1Eg + 2Eg",
                          "Ag",
                          "Ag + 1Eg + 2Eg",
                          "1Eg+ 2Eg",
                          "Ag",
                          "Ag + 1Eg + 2Eg",
                          "1Eg + 2Eg",
                          "Ag + 1Eg + 2Eg"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[3][bdDir[7]]+RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["1Eg + 2Eg",  # x(yx)y
                           "1Eg + 2Eg", # x(yz)y
                           "1Eg + 2Eg",  # x(zx)y
                           "Ag",  # x(zz)y
                           "1Eg + 2Eg", # x(yx)z
                           "Ag + 1Eg + 2Eg",  # x(yy)z
                           "1Eg + 2Eg", # x(zx)z
                           "1Eg + 2Eg", # x(zy)z
                           "Ag + 1Eg + 2Eg", # y(xx)z
                           "1Eg + 2Eg",  # y(xy)z
                           "1Eg + 2Eg", # y(zx)z
                           "1Eg + 2Eg"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]]+RTs[5][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]]+RTs[5][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]]+RTs[5][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]]+RTs[5][rDir[8]],\
                           RTs[3][rDir[9]]+RTs[5][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
    
    elif pointgroup == "32":
        backscattering = ["A1 + E(LO)",
                          "E(LO)",
                          "A1",
                          "A1 + E(TO)",
                          "E(LO)",
                          "A1",
                          "A1 + E(TO)",
                          "E(TO)",
                          "A1 + E(TO)"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[3][bdDir[7]]+RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["E(LO+TO)",  # x(yx)y
                           "E(LO+TO)", # x(yz)y
                           "E(LO+TO)",  # x(zx)y
                           "A1",  # x(zz)y
                           "E(TO)", # x(yx)z
                           "A1 + E(LO+TO)",  # x(yy)z
                           "E(TO)", # x(zx)z
                           "E(LO+TO)", # x(zy)z
                           "A1 + E(TO)", # y(xx)z
                           "E(LO+TO)",  # y(xy)z
                           "E(LO+TO)", # y(zx)z
                           "E(TO)"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]]+RTs[5][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]]+RTs[5][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]]+RTs[5][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]]+RTs[5][rDir[8]],\
                           RTs[3][rDir[9]]+RTs[5][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
        
    elif pointgroup == "3m":
        backscattering = ["A1(TO) + E(TO)",
                          "E(TO)",
                          "A1(TO)",
                          "A1(TO) + E(LO)",
                          "E(TO)",
                          "A1(TO)",
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

    elif pointgroup == "-3m":
        backscattering = ["A1g + Eg",
                          "Eg",
                          "A1g",
                          "A1g + Eg",
                          "Eg",
                          "A1g",
                          "A1g + Eg",
                          "Eg",
                          "A1g + Eg"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[3][bdDir[7]]+RTs[5][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["Eg",  # x(yx)y
                           "Eg", # x(yz)y
                           "Eg",  # x(zx)y
                           "A1g",  # x(zz)y
                           "Eg", # x(yx)z
                           "A1g + Eg",  # x(yy)z
                           "Eg", # x(zx)z
                           "Eg", # x(zy)z
                           "A1g + Eg", # y(xx)z
                           "Eg",  # y(xy)z
                           "Eg", # y(zx)z
                           "Eg"] # y(zy)z
        rightComponents = [RTs[3][rDir[0]]+RTs[5][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[3][rDir[4]]+RTs[5][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]]+RTs[5][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]]+RTs[5][rDir[8]],\
                           RTs[3][rDir[9]]+RTs[5][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
    
    elif pointgroup == "6":
        backscattering = ["A(TO) + 1E2 + 2E2",
                          "1E1(LO) + 2E1(TO)",
                          "A(TO)",
                          "A(TO) + 1E2 + 2E2",
                          "1E1(TO) + 2E1(LO)",
                          "A(TO)",
                          "A(LO) + 1E2 + 2E2",
                          "1E2 + 2E2",
                          "A(LO) + 1E2 + 2E2"]
        backComponents = [RTs[1][bdDir[0]]+RTs[7][bdDir[0]]+RTs[9][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[7][bdDir[3]]+RTs[9][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[7][bdDir[6]]+RTs[9][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[7][bdDir[8]]+RTs[9][bdDir[8]]]
        rightscattering = ["1E2 + 2E2",  # x(yx)y
                           "1E1(LO+TO) + 2E1(LO+TO)", # x(yz)y
                           "1E1(LO+TO) + 2E1(LO+TO)",  # x(zx)y
                           "A(TO)",  # x(zz)y
                           "1E2 + 2E2", # x(yx)z
                           "A(LO+TO) + 1E2 + 2E2",  # x(yy)z
                           "1E1(LO+TO) + 2E1(TO)", # x(zx)z
                           "1E1(LO+TO) + 2E1(TO)", # x(zy)z
                           "A(LO+TO) + 1E2 + 2E2", # y(xx)z
                           "1E2 + 2E2",  # y(xy)z
                           "1E1(TO) + 2E1(LO+TO)", # y(zx)z
                           "1E1(TO) + 2E1(LO+TO)"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[7][rDir[5]]+RTs[9][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[7][rDir[8]]+RTs[9][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
    
    elif pointgroup == "-6":
        backscattering = ["A' + 1E' + 2E'",
                          "1E'' + 2E''",
                          "A'",
                          "A' + 1E' + 2E'",
                          "1E'' + 2E''",
                          "A'",
                          "A' + 1E' + 2E'",
                          "1E' + 2E'",
                          "A' + 1E' + 2E'"]
        backComponents = [RTs[1][bdDir[0]]+RTs[7][bdDir[0]]+RTs[9][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[7][bdDir[3]]+RTs[9][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[7][bdDir[6]]+RTs[9][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[7][bdDir[8]]+RTs[9][bdDir[8]]]
        rightscattering = ["1E' + 2E'",  # x(yx)y
                           "1E'' + 2E''", # x(yz)y
                           "1E'' + 2E''",  # x(zx)y
                           "A'",  # x(zz)y
                           "1E' + 2E'", # x(yx)z
                           "A' + 1E' + 2E'",  # x(yy)z
                           "1E'' + 2E''", # x(zx)z
                           "1E'' + 2E''", # x(zy)z
                           "A' + 1E' + 2E'", # y(xx)z
                           "1E' + 2E'",  # y(xy)z
                           "1E'' + 2E''", # y(zx)z
                           "1E'' + 2E''"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[7][rDir[5]]+RTs[9][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[7][rDir[8]]+RTs[9][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
    
    elif pointgroup == "6/m":
        backscattering = ["Ag + 1E2g + 2E2g",
                          "1E1g + 2E1g",
                          "Ag",
                          "Ag + 1E2g + 2E2g",
                          "1E1g + 2E1g",
                          "Ag",
                          "Ag + 1E2g + 2E2g",
                          "1E2g + 2E2g",
                          "Ag + 1E2g + 2E2g"]
        backComponents = [RTs[1][bdDir[0]]+RTs[7][bdDir[0]]+RTs[9][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[7][bdDir[3]]+RTs[9][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[7][bdDir[6]]+RTs[9][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[7][bdDir[8]]+RTs[9][bdDir[8]]]
        rightscattering = ["1E2g + 2E2g",  # x(yx)y
                           "1E1g + 2E1g", # x(yz)y
                           "1E1g + 2E1g",  # x(zx)y
                           "Ag",  # x(zz)y
                           "1E2g + 2E2g", # x(yx)z
                           "Ag + 1E2g + 2E2g",  # x(yy)z
                           "1E1g + 2E1g", # x(zx)z
                           "1E1g + 2E1g", # x(zy)z
                           "Ag + 1E2g + 2E2g", # y(xx)z
                           "1E2g + 2E2g",  # y(xy)z
                           "1E1g + 2E1g", # y(zx)z
                           "1E1g + 2E1g"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[7][rDir[5]]+RTs[9][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[7][rDir[8]]+RTs[9][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
        
    elif pointgroup == "622":
        backscattering = ["A1 + E2",
                          "E1(LO)",
                          "A1",
                          "A1 + E2",
                          "E1(LO)",
                          "A1",
                          "A1 + E2",
                          "E2",
                          "A1 + E2"]
        backComponents = [RTs[1][bdDir[0]]+RTs[7][bdDir[0]]+RTs[9][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[7][bdDir[3]]+RTs[9][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[7][bdDir[6]]+RTs[9][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[7][bdDir[8]]+RTs[9][bdDir[8]]]
        rightscattering = ["E2",  # x(yx)y
                           "E1(LO+TO)", # x(yz)y
                           "E1(LO+TO)",  # x(zx)y
                           "A1",  # x(zz)y
                           "E2", # x(yx)z
                           "A1 + E2",  # x(yy)z
                           "E1(TO)", # x(zx)z
                           "E1(LO+TO)", # x(zy)z
                           "A1 + E2", # y(xx)z
                           "E2",  # y(xy)z
                           "E1(LO+TO)", # y(zx)z
                           "E1(TO)"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[7][rDir[5]]+RTs[9][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[7][rDir[8]]+RTs[9][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
    
    elif pointgroup == "6mm":
        backscattering = ["A1(TO) + E2",
                          "E1(TO)",
                          "A1(TO)",
                          "A1(TO) + E2",
                          "E1(TO)",
                          "A1(TO)",
                          "A1(LO) + E2",
                          "E2",
                          "A1(LO) + E2"]
        backComponents = [RTs[1][bdDir[0]]+RTs[7][bdDir[0]]+RTs[9][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[7][bdDir[3]]+RTs[9][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[7][bdDir[6]]+RTs[9][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[7][bdDir[8]]+RTs[9][bdDir[8]]]
        rightscattering = ["E2",  # x(yx)y
                           "E1(LO+TO)", # x(yz)y
                           "E1(LO+TO)",  # x(zx)y
                           "A1(TO)",  # x(zz)y
                           "E2", # x(yx)z
                           "A1(LO+TO) + E2",  # x(yy)z
                           "E1(LO+TO)", # x(zx)z
                           "E1(TO)", # x(zy)z
                           "A1(LO+TO) + E2", # y(xx)z
                           "E2",  # y(xy)z
                           "E1(TO)", # y(zx)z
                           "E1(LO+TO)"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[7][rDir[5]]+RTs[9][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[7][rDir[8]]+RTs[9][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
    
    elif pointgroup == "-62m":
        backscattering = ["A1' + E'(LO)",
                          "E''",
                          "A1'",
                          "A1' + E'(TO)",
                          "E''",
                          "A1'",
                          "A1' + E'(TO)",
                          "E'(TO)",
                          "A1' + E'(TO)"]
        backComponents = [RTs[1][bdDir[0]]+RTs[7][bdDir[0]]+RTs[9][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[7][bdDir[3]]+RTs[9][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[7][bdDir[6]]+RTs[9][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[7][bdDir[8]]+RTs[9][bdDir[8]]]
        rightscattering = ["E'(LO+TO)",  # x(yx)y
                           "E''", # x(yz)y
                           "E''",  # x(zx)y
                           "A1'",  # x(zz)y
                           "E'(TO)", # x(yx)z
                           "A1'' + E'(LO+TO)",  # x(yy)z
                           "E''", # x(zx)z
                           "E''", # x(zy)z
                           "A1' + E'(TO)", # y(xx)z
                           "E'(LO+TO)",  # y(xy)z
                           "E''", # y(zx)z
                           "E''"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[7][rDir[5]]+RTs[9][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[7][rDir[8]]+RTs[9][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
    
    elif pointgroup == "6/mmm":
        backscattering = ["A1g + E2g",
                          "E1g",
                          "A1g",
                          "A1g + E2g",
                          "E1g",
                          "A1g",
                          "A1g + E2g",
                          "E2g",
                          "A1g + E2g"]
        backComponents = [RTs[1][bdDir[0]]+RTs[7][bdDir[0]]+RTs[9][bdDir[0]],\
                          RTs[3][bdDir[1]]+RTs[5][bdDir[1]],\
                          RTs[1][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[7][bdDir[3]]+RTs[9][bdDir[3]],\
                          RTs[3][bdDir[4]]+RTs[5][bdDir[4]],\
                          RTs[1][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[7][bdDir[6]]+RTs[9][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]],\
                          RTs[1][bdDir[8]]+RTs[7][bdDir[8]]+RTs[9][bdDir[8]]]
        rightscattering = ["E2g",  # x(yx)y
                           "E1g", # x(yz)y
                           "E1g",  # x(zx)y
                           "A1g",  # x(zz)y
                           "E2g", # x(yx)z
                           "A1g + E2g",  # x(yy)z
                           "E1g", # x(zx)z
                           "E1g", # x(zy)z
                           "A1g + E2g", # y(xx)z
                           "E2g",  # y(xy)z
                           "E1g", # y(zx)z
                           "E1g"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[7][rDir[5]]+RTs[9][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[7][rDir[8]]+RTs[9][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]],\
                           RTs[3][rDir[10]]+RTs[5][rDir[10]],\
                           RTs[3][rDir[11]]+RTs[5][rDir[11]]]
    
    elif pointgroup == "23":
        backscattering = ["A + 1E + 2E",
                          "T(LO)",
                          "A + 1E",
                          "A + 1E + 2E",
                          "T(LO)",
                          "A + 1E",
                          "A + 1E + 2E",
                          "T(LO)",
                          "A + 1E + 2E"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[7][bdDir[1]]+RTs[9][bdDir[1]]+RTs[11][bdDir[1]],\
                          RTs[1][bdDir[2]]+RTs[3][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[7][bdDir[4]]+RTs[9][bdDir[4]]+RTs[11][bdDir[4]],\
                          RTs[1][bdDir[5]]+RTs[3][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]]+RTs[11][bdDir[7]],
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["T(TO)", # x(yx)y
                           "T(LO+TO)",  # x(yz)y
                           "T(LO+TO)", # x(zx)y
                           "A + 1E", # x(zz)y
                           "T(LO+TO)",  # x(yx)z
                           "A + 1E + 2E", # x(yy)z
                           "T(TO)", # x(zx)z
                           "T(LO+TO)", # x(zy)z
                           "A + 1E + 2E", # y(xx)z
                           "T(LO+TO)",  # y(xy)z
                           "T(LO+TO)", # y(xy)z
                           "T(TO)"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]]+RTs[11][rDir[0]],\
                           RTs[7][rDir[1]]+RTs[9][rDir[1]]+RTs[11][rDir[1]],\
                           RTs[7][rDir[2]]+RTs[9][rDir[2]]+RTs[11][rDir[2]],\
                           RTs[1][rDir[3]]+RTs[3][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]]+RTs[11][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]]+RTs[5][rDir[5]],\
                           RTs[7][rDir[6]]+RTs[9][rDir[6]]+RTs[11][rDir[6]],\
                           RTs[7][rDir[7]]+RTs[9][rDir[7]]+RTs[11][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]]+RTs[5][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]]+RTs[11][rDir[9]],\
                           RTs[7][rDir[10]]+RTs[9][rDir[10]]+RTs[11][rDir[10]],\
                           RTs[7][rDir[11]]+RTs[9][rDir[11]]+RTs[11][rDir[11]]]
    
    elif pointgroup == "m-3":
        backscattering = ["Ag + 1Eg + 2Eg",
                          "Tg",
                          "Ag + 1Eg",
                          "Ag + 1Eg + 2Eg",
                          "Tg",
                          "Ag + 1Eg",
                          "Ag + 1Eg + 2Eg",
                          "Tg",
                          "Ag + 1Eg + 2Eg"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[7][bdDir[1]]+RTs[9][bdDir[1]]+RTs[11][bdDir[1]],\
                          RTs[1][bdDir[2]]+RTs[3][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[7][bdDir[4]]+RTs[9][bdDir[4]]+RTs[11][bdDir[4]],\
                          RTs[1][bdDir[5]]+RTs[3][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]]+RTs[11][bdDir[7]],
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["Tg", # x(yx)y
                           "Tg",  # x(yz)y
                           "Tg", # x(zx)y
                           "Ag + 1Eg", # x(zz)y
                           "Tg",  # x(yx)z
                           "Ag + 1Eg + 2Eg", # x(yy)z
                           "Tg", # x(zx)z
                           "Tg", # x(zy)z
                           "Ag + 1Eg + 2Eg", # y(xx)z
                           "Tg",  # y(xy)z
                           "Tg", # y(xy)z
                           "Tg"] # y(zy)z
        rightComponents = [RTs[7][rDir[0]]+RTs[9][rDir[0]]+RTs[11][rDir[0]],\
                           RTs[7][rDir[1]]+RTs[9][rDir[1]]+RTs[11][rDir[1]],\
                           RTs[7][rDir[2]]+RTs[9][rDir[2]]+RTs[11][rDir[2]],\
                           RTs[1][rDir[3]]+RTs[3][rDir[3]],\
                           RTs[7][rDir[4]]+RTs[9][rDir[4]]+RTs[11][rDir[4]],\
                           RTs[1][rDir[5]]+RTs[3][rDir[5]]+RTs[5][rDir[5]],\
                           RTs[7][rDir[6]]+RTs[9][rDir[6]]+RTs[11][rDir[6]],\
                           RTs[7][rDir[7]]+RTs[9][rDir[7]]+RTs[11][rDir[7]],\
                           RTs[1][rDir[8]]+RTs[3][rDir[8]]+RTs[5][rDir[8]],\
                           RTs[7][rDir[9]]+RTs[9][rDir[9]]+RTs[11][rDir[9]],\
                           RTs[7][rDir[10]]+RTs[9][rDir[10]]+RTs[11][rDir[10]],\
                           RTs[7][rDir[11]]+RTs[9][rDir[11]]+RTs[11][rDir[11]]]
    
    elif pointgroup == "432":
        backscattering = ["A1 + E",
                          "T2",
                          "A1 + E",
                          "A1 + E",
                          "T2",
                          "A1 + E",
                          "A1 + E",
                          "T2",
                          "A1 + E"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[7][bdDir[1]]+RTs[9][bdDir[1]]+RTs[11][bdDir[1]],\
                          RTs[1][bdDir[2]]+RTs[3][bdDir[2]]+RTs[5][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[7][bdDir[4]]+RTs[9][bdDir[4]]+RTs[11][bdDir[4]],\
                          RTs[1][bdDir[5]]+RTs[3][bdDir[5]]+RTs[5][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]]+RTs[11][bdDir[7]],
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["T2", # x(yx)y
                           "T2",  # x(yz)y
                           "T2", # x(zx)y
                           "A1 + E", # x(zz)y
                           "T2",  # x(yx)z
                           "A1 + E", # x(yy)z
                           "T2", # x(zx)z
                           "T2", # x(zy)z
                           "A1 + E", # y(xx)z
                           "T2",  # y(xy)z
                           "T2", # y(xy)z
                           "T2"] # y(zy)z
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
    
    elif pointgroup == "-43m":
        backscattering = ["A1 + E",
                          "T2(LO)",
                          "A1 + E",
                          "A1 + E",
                          "T2(LO)",
                          "A1 + E",
                          "A1 + E",
                          "T2(LO)",
                          "A1 + E"]
        backComponents = [RTs[1][bdDir[0]]+RTs[3][bdDir[0]]+RTs[5][bdDir[0]],\
                          RTs[7][bdDir[1]]+RTs[9][bdDir[1]]+RTs[11][bdDir[1]],\
                          RTs[1][bdDir[2]]+RTs[3][bdDir[2]]+RTs[5][bdDir[2]],\
                          RTs[1][bdDir[3]]+RTs[3][bdDir[3]]+RTs[5][bdDir[3]],\
                          RTs[7][bdDir[4]]+RTs[9][bdDir[4]]+RTs[11][bdDir[4]],\
                          RTs[1][bdDir[5]]+RTs[3][bdDir[5]]+RTs[5][bdDir[5]],\
                          RTs[1][bdDir[6]]+RTs[3][bdDir[6]]+RTs[5][bdDir[6]],\
                          RTs[7][bdDir[7]]+RTs[9][bdDir[7]]+RTs[11][bdDir[7]],
                          RTs[1][bdDir[8]]+RTs[3][bdDir[8]]+RTs[5][bdDir[8]]]
        rightscattering = ["T2(TO)", # x(yx)y
                           "T2(LO+TO)",  # x(yz)y
                           "T2(LO+TO)", # x(zx)y
                           "A1 + E", # x(zz)y
                           "T2(LO+TO)",  # x(yx)z
                           "A1 + E", # x(yy)z
                           "T2(TO)", # x(zx)z
                           "T2(LO+TO)", # x(zy)z
                           "A1 + E", # y(xx)z
                           "T2(LO+TO)",  # y(xy)z
                           "T2(LO+TO)", # y(xy)z
                           "T2(TO)"] # y(zy)z
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
        return [], [], [], []
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


# get the tensor information to printable output
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

def findInListOfList(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return (mylist.index(sub_list), sub_list.index(char))
    raise ValueError("'{char}' is not in list".format(char = char))
#

#############
# phonon symmetry analysis
#############

def getIrrepsSymbols(path, basis, coord, elements, symprec, degeneracy_tolerance):
    pwd = os.getcwd()
    os.chdir(path)

    cell = PhonopyAtoms(
        symbols=elements,
        cell=basis,
        scaled_positions=coord,
    )

    phonopy_instance = phonopy.load(
        unitcell=cell,
        supercell_matrix=np.eye(3),
        primitive_matrix="auto",
        force_constants_filename="FORCE_CONSTANTS",
    )

    os.chdir(pwd)

    ir = IrReps(
        phonopy_instance.dynamical_matrix,
        np.array([0.0, 0.0, 0.0], dtype=float),
        phonopy_instance.primitive_symmetry,
        symprec=symprec,
        degeneracy_tolerance=degeneracy_tolerance
    )

    ir.run()

    tmp_labels = ir._get_ir_labels()

    labels = []
    for j in range(len(ir._degenerate_sets)):
        for i in range(len(ir._degenerate_sets[j])):
            labels.append(tmp_labels[j])
        #
    #

    return labels, ir._pointgroup_symbol
#

###########
# phonon analysis for Raman
###########

def getAcoustics(modelist, eigvecs, eigvals, masses): 
    # get the candidates for possible acoustic modes
    candidates = []
    for mode in modelist:
        if np.abs(eigvals[mode]) < 10:
            candidates.append(mode)
        #
    #   
    masses = np.array(masses)
    sqrt_m = np.sqrt(masses)
    
    indicators = {}
    for mode in modelist:
        eigvec = np.array(eigvecs[mode])
        S = np.sum(sqrt_m[:, None] * eigvec, axis=0) 
        indicators[mode] = np.linalg.norm(S)
    #

    top3_keys = heapq.nlargest(3, indicators, key=indicators.get)

    acoustics = []
    for mode in candidates:
        if mode in top3_keys and mode in candidates and len(acoustics) < 3:
            acoustics.append(mode)
        #
    #
    
    if len(acoustics) > 0:
        return acoustics
    else:
        print("[getAcoustics]: Could not determine acoustic modes")
        return []
    #
#

def getRotations(modelist, masses, positions, eigvecs, tol=0.8):

    def mw_dot(a, b):
        return np.sum(masses[:, None] * a * b)

    M = np.sum(masses)
    r_cm = np.sum(positions * masses[:, None], axis=0) / M

    rotations = []

    for j in modelist:
        eigvec = eigvecs[j]
        A = []
        b = []
        for m, r, u in zip(masses, positions, eigvec):
            dr = r - r_cm
            C = np.array([
                        [0,      -dr[2],  dr[1]],
                        [dr[2],   0,     -dr[0]],
                        [-dr[1],  dr[0],  0    ]
                        ])
            # mass-weighted least squares
            A.append(np.sqrt(m) * C)
            b.append(np.sqrt(m) * u)
        A = np.vstack(A)
        b = np.vstack(b).reshape(-1)

        omega, *_ = np.linalg.lstsq(A, b, rcond=None)

        u_rot = np.cross(omega, positions - r_cm)

        num = mw_dot(eigvec, u_rot)
        den = np.sqrt(mw_dot(eigvec, eigvec) * mw_dot(u_rot, u_rot))

        overlap = np.abs(num / den) if den > 0 else 0.0

        if overlap > tol:
            rotations.append(j)
        #

    return np.array([x+1 for x in rotations])
#

def getDegenerates(modelist, eigvals, labels, prec=1e0):
    degenerates = []
    for j in modelist:
        for k in modelist:
            if j < k:
                tmp = []
                if np.abs(eigvals[j] - eigvals[k]) < prec and labels[j] == labels[k]:
                    if j not in tmp:
                        tmp.append(j)
                    #
                    tmp.append(k)
                #
                if tmp != []:
                    degenerates.append(tmp)
                #
            #
        #
        if j not in flatten(degenerates):
            degenerates.append([j])
    #
    return degenerates
#

def getRamanSilent(modelist, labels, pointgroup):
    RamanTensors = RamanTensorComponents[pointgroup]
    silent = []
    for mode in modelist:
        if labels[mode] not in RamanTensors[::2]:
            silent.append(mode)
        #
    #
    return silent
#

def getRamanSilentOvertones(pointgroup, ramantensors, modelist, labels):
    chartable = CHAR_TABLES[HM_TO_SCHOENFLIES[pointgroup]]
    class_sizes = []
    for label in chartable["classes"]:
        m = re.match(r"(\d+)", label)
        class_sizes.append(int(m.group(1)) if m else 1)
    #
    class_sizes = np.array(class_sizes)
    group_order = class_sizes.sum()
    modelist2nd = []

    for mode1 in modelist:
        for mode2 in modelist:
            if mode1 <= mode2:
                chi_prod = np.array(chartable["irreps"][labels[mode1]]) * np.array(chartable["irreps"][labels[mode2]])
                multiplicities = {}
                for name, chi in chartable["irreps"].items():
                    n = np.sum(class_sizes * chi * chi_prod) / group_order
                    multiplicities[name] = int(round(n))
                #
                for label in multiplicities.keys():
                    if label in ramantensors[::2] and multiplicities[label] > 0:
                        modelist2nd.append([mode1, mode2])
                        break
                    #
                #
            #
        #
    #
    return modelist2nd
#

############
# get tensors and selection rules
###########

def analyzeRamanTensors(pointgroup, varprint=False):
    RamanTensors = RamanTensorComponents[pointgroup]

    # print to console
    if varprint == True:
        print("Raman Tensors of point group "+pointgroup)
        for i in range(int(len(RamanTensors)/2)):
            print(RamanTensors[2*i])
            for j in range(3):
                print(RamanTensors[2*i+1][j])
            #
        #
        print("")
    #
    return RamanTensors
#

def analyzeDielectricTensor(pointgroup, varprint=False):
    dielectricTensor = dielectricFunctionComponents[pointgroup]

    # print to console
    if varprint == True:
        print("Dielectric Tensor of point group "+pointgroup)
        for j in range(3):
            print(dielectricTensor[j])
        #
        print("")
    #
    return dielectricTensor
#

def RamanSelection(pointgroup, RamanTensors):
    backscattering, backComponents, rightscattering, rightComponents = RamanSelectionRules(pointgroup, RamanTensors)

    # format the components
    for j in range(len(backComponents)):
        backComponents[j] = formatString(backComponents[j])
    #
    for j in range(len(rightComponents)):
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
    scattering = IRSelectionRules[pointgroup]

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

def getDecomposition(labels):
    decomposition = {}
    for i in set(labels):
       decomposition[i] = labels.count(i)
    #
    printstr = ""
    for label, number in decomposition.items():
        if printstr == "":
            printstr += "Γ = {}({})".format(number, label)
        else:
            printstr += " + {}({})".format(number, label)
        #
    #
    print("Mode decomposition at Γ")
    print(printstr)
    print("")
#
