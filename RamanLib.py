#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import sys
import numpy as np

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

backDirs = ["x(yy)x\u0305", "x(yz)x\u0305", "x(zz)x\u0305", "y(xx)y\u0305", "y(xz)y\u0305", "y(zz)y\u0305", "z(xx)z\u0305", "z(xy)z\u0305", "z(yy)z\u0305"]
bdDir = {0: (1,1), 1: (1,2), 2: (2,2), 3: (0,0), 4: (0,2), 5: (2,2), 6: (0,0), 7: (0,1), 8: (1,1)}
rightDirs = ["x(yx)y", "x(yz)y", "x(zx)y", "x(zz)y", "y(xx)z", "y(xy)z", "y(zx)z", "y(zy)z"]
rDir = {0: (0,1), 1: (1,2), 2: (0,2), 3: (2,2), 4: (0,0), 5:(0,1), 6: (0,2), 7: (1,2)}

def RamanTensorComponents(pointgroup):
    if pointgroup == "1":
        A = np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype ="str")
        RamanTensors = ["A", A]
    
    elif pointgroup == "-1":
        Ag = np.array([["a", "d", "e"], ["d", "b", "f"], ["e", "f", "c"]], dtype ="str")
        RamanTensors = ["Ag", Ag]

    elif pointgroup == "2":
        A = np.array([["a", "d", 0], ["d", "b", 0], [0, 0, "c"]], dtype ="str")
        Bxy = np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype ="str")
        RamanTensors = ["A", A, "B(x,y)", Bxy]

    elif pointgroup == "m":
        Axy = np.array([["a", "d", 0], ["d", "b", 0], [0, 0, "c"]], dtype ="str")
        Az = np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype ="str")
        RamanTensors = ["A'(x,y)", Axy, "A''(z)", Az]

    elif pointgroup == "2/m":
        Ag = np.array([["a", "d", 0], ["d", "b", 0], [0, 0, "c"]], dtype ="str")
        Bg = np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype ="str")
        RamanTensors = ["Ag", Ag, "Bg", Bg]

    elif pointgroup == "222":
        A = np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype ="str")
        B1z = np.array([[0, "d", 0], [0, "d", 0], [0, 0, 0]], dtype ="str")
        B2y = np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype ="str")
        B3x = np.array([[0, 0, 0], [0, 0, "f"], [0, "f", 0]], dtype ="str")
        RamanTensors = ["A", A, "B1(z)", B1z, "B2(y)", B2y, "B3(x)", B3x]

    elif pointgroup == "mm2":
        A1z = np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype ="str")
        A2 = np.array([[0, "d", 0], [0, "d", 0], [0, 0, 0]], dtype ="str")
        B1x = np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype ="str")
        B2y = np.array([[0, 0, 0], [0, 0, "f"], [0, "f", 0]], dtype ="str")
        RamanTensors = ["A1(z)", A1z, "A2", A2, "B1(x)", B1x, "B2(y)", B2y]

    elif pointgroup == "mmm":
        Ag = np.array([["a", 0, 0], [0, "b", 0], [0, 0, "c"]], dtype ="str")
        B1g = np.array([[0, "d", 0], [0, "d", 0], [0, 0, 0]], dtype ="str")
        B2g = np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype ="str")
        B3g = np.array([[0, 0, 0], [0, 0, "f"], [0, "f", 0]], dtype ="str")
        RamanTensors = ["Ag", Ag, "B1g", B1g, "B2g", B2g, "B3g", B3g]

    elif pointgroup == "4":
        Az = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        B = np.array([["c", "d", 0], ["d", "-c", 0], [0, 0, 0]], dtype ="str")
        E1x = np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype ="str")
        E2y = np.array([[0, 0, "-f"], [0, 0, "e"], ["-f", "e", 0]], dtype ="str")
        RamanTensors = ["A(z)", Az, "B", B, "1E(x)", E1x, "2Ey", E2y]

    elif pointgroup == "-4":
        Az = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        B = np.array([["c", "d", 0], ["d", "-c", 0], [0, 0, 0]], dtype ="str")
        E1x = np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype ="str")
        E2y = np.array([[0, 0, "f"], [0, 0, "-e"], ["f", "-e", 0]], dtype ="str")
        RamanTensors = ["A(z)", Az, "B(z)", B, "1E(x)", E1x, "2Ey", E2y]

    elif pointgroup == "4/m":
        Ag = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        Bg = np.array([["c", "d", 0], ["d", "-c", 0], [0, 0, 0]], dtype ="str")
        E1g = np.array([[0, 0, "e"], [0, 0, "f"], ["e", "f", 0]], dtype ="str")
        E2g = np.array([[0, 0, "-f"], [0, 0, "e"], ["-f", "e", 0]], dtype ="str")
        RamanTensors = ["Ag", Ag, "Bg", Bg, "1Eg", E1g, "2Eg", E2g]
    
    elif pointgroup == "4222":
        A1 = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        B1 = np.array([["c", 0, 0], [0, "-c", 0], [0, 0, 0]], dtype ="str")
        B2 = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        Ex = np.array([[0, 0, 0], [0, 0, "e"], [0, "e", 0]], dtype ="str")
        Ey = np.array([[0, 0, "-e"], [0, 0, 0], ["-e", 0, 0]], dtype ="str")
        RamanTensors = ["A1", A1, "B1", B1, "B2", B2, "E(x)", Ex, "E(y)", Ey]

    elif pointgroup == "4222":
        Az = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        B1 = np.array([["c", 0, 0], [0, "-c", 0], [0, 0, 0]], dtype ="str")
        B2 = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        Ex = np.array([[0, 0, 0], [0, 0, "e"], [0, "e", 0]], dtype ="str")
        Ey = np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype ="str")
        RamanTensors = ["A(z)", Az, "B1", B1, "B2", B2, "E(x)", Ex, "E(y)", Ey]

    elif pointgroup == "-42m":
        A1 = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        B1 = np.array([["c", 0, 0], [0, "-c", 0], [0, 0, 0]], dtype ="str")
        B2z = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        Ex = np.array([[0, 0, 0], [0, 0, "e"], [0, "e", 0]], dtype ="str")
        Ey = np.array([[0, 0, "e"], [0, 0, 0], ["e", 0, 0]], dtype ="str")
        RamanTensors = ["A1", A1, "B1", B1, "B2(z)", B2z, "E(x)", Ex, "E(y)", Ey]

    elif pointgroup == "4/mmm":
        A1g = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        B1g = np.array([["c", 0, 0], [0, "-c", 0], [0, 0, 0]], dtype ="str")
        B2g = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        Eg = np.array([[0, 0, 0], [0, 0, "e"], [0, "e", 0]], dtype ="str")
        Eg2 = np.array([[0, 0, "-e"], [0, 0, 0], ["-e", 0, 0]], dtype ="str")
        RamanTensors = ["A1g", A1g, "B1g", B1g, "B2g", B2g, "Eg", Eg, "Eg", Eg2]

    elif pointgroup == "3":
        Az = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype="str")
        E1x = np.array([["c", "d", "e"], ["d", "-c", "f"], ["e", "f", 0]], dtype="str")
        E2y = np.array([["d", "-c", "-f"], ["-c", "-d", "e"], ["e", "-f", 0]], dtype="str")
        RamanTensors = ["A(z)", Az, "1E(x)", E1x, "2E(y)", E2y]

    elif pointgroup == "-3":
        Ag = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype="str")
        E1x = np.array([["c", "d", "e"], ["d", "-c", "f"], ["e", "f", 0]], dtype="str")
        E2y = np.array([["d", "-c", "-f"], ["-c", "-d", "e"], ["e", "-f", 0]], dtype="str")
        RamanTensors = ["Ag", Az, "1E(x)", E1x, "2E(y)", E2y]

    elif pointgroup == "32":
        A1 = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype="str")
        Ex = np.array([["c", 0, 0], [0, "-c", "d"], [0, "d", 0]], dtype="str")
        Ey = np.array([[0, "-c", "-d"], ["-c", 0, 0], ["-d", 0, 0]], dtype="str")
        RamanTensors = ["A1", A1, "Ex", Ex, "Ey", Ey]

    elif pointgroup == "3m":
        A1 = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype="str")
        Ex = np.array([[0, "c", "d"], ["c", 0, 0], ["d", 0, 0]], dtype="str")
        Ey = np.array([["c", 0, 0], [0, "-c", "d"], [0, "d", 0]], dtype="str")
        RamanTensors = ["A1", A1, "Ex", Ex, "Ey", Ey]

    elif pointgroup == "-3m":
        A1g = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype="str")
        Eg1 = np.array([["c", 0, 0], [0, "-c", "d"], [0, "d", 0]], dtype="str")
        Eg2 = np.array([[0, "-c", "-d"], ["-c", 0, 0], ["-d", 0, 0]], dtype="str")
        RamanTensors = ["A1g", A1g, "Eg,1", Eg1, "Eg,2", Eg2]    
    
    elif pointgroup == "6":
        Az = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        E11x = np.array([[0, 0, "c"], [0, 0, "d"], ["c", "d", 0]], dtype ="str")
        E12y = np.array([[0, 0, "-d"], [0, 0, "c"], ["-d", "c", 0]], dtype ="str")
        E12 = np.array([["e", "f", 0], ["f", "-e", 0], [0, 0, 0]], dtype ="str")
        E22 = np.array([["f", "-e", 0], ["-e", "-f", 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A(z)", Az, "1E1(x)", E11x, "2E1(y)", E12y, "1E2", E12, "2E2", E22]

    elif pointgroup == "-6":
        A = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        E1 = np.array([[0, 0, "c"], [0, 0, "d"], ["c", "d", 0]], dtype ="str")
        E2 = np.array([[0, 0, "-d"], [0, 0, "c"], ["-d", "c", 0]], dtype ="str")
        E1x = np.array([["e", "f", 0], ["f", "-e", 0], [0, 0, 0]], dtype ="str")
        E2y = np.array([["f", "-e", 0], ["-e", "-f", 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A'", Az, "1E''", E1, "2E''", E2, "1E'x", E1x, "2E'y", E2y]

    elif pointgroup == "6/m":
        Ag = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        E11g = np.array([[0, 0, "c"], [0, 0, "d"], ["c", "d", 0]], dtype ="str")
        E21g = np.array([[0, 0, "-d"], [0, 0, "c"], ["-d", "c", 0]], dtype ="str")
        E12g = np.array([["e", "f", 0], ["f", "-e", 0], [0, 0, 0]], dtype ="str")
        E22g = np.array([["f", "-e", 0], ["-e", "-f", 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A'", Az, "1E1g", E11g, "2E1g", E21g, "1E2g", E12g, "2E2g", E22g]

    elif pointgroup == "622":
        Az = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        E1x = np.array([[0, 0, 0], [0, 0, "c"], [0, "c", 0]], dtype ="str")
        E1y = np.array([[0, 0, "-c"], [0, 0, 0], ["-c", 0, 0]], dtype ="str")
        E21 = np.array([["d", 0, 0], [0, "-d", 0], [0, 0, 0]], dtype ="str")
        E22 = np.array([[0, "-d", 0], ["-d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A1", Az, "E1(x)", E1x, "E1(y)", E1y, "E2", E21, "E2", E22]

    elif pointgroup == "6mm":
        A1z = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        E1y = np.array([[0, 0, "c"], [0, 0, 0], ["c", 0, 0]], dtype ="str")
        E1x = np.array([[0, 0, 0], [0, 0, "c"], [0, "c", 0]], dtype ="str")
        E21 = np.array([["d", 0, 0], [0, "-d", 0], [0, 0, 0]], dtype ="str")
        E22 = np.array([[0, "-d", 0], ["-d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A1(z)", A1z, "E1(x)", E1x, "E1(y)", E1y, "E2", E21, "E2", E22]

    elif pointgroup == "-62m":
        A1 = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        E2 = np.array([[0, 0, 0], [0, 0, "c"], [0, "c", 0]], dtype ="str")
        E1 = np.array([[0, 0, "-c"], [0, 0, 0], ["-c", 0, 0]], dtype ="str")
        Ex = np.array([["d", 0, 0], [0, "-d", 0], [0, 0, 0]], dtype ="str")
        Ey = np.array([[0, "-d", 0], ["-d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A'1", A1, "E''", E1x, "E''", E1y, "E'(x)", E21, "E'(y)", E22]

    elif pointgroup == "6/mmm":
        A1g = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "b"]], dtype ="str")
        E11g = np.array([[0, 0, 0], [0, 0, "c"], [0, "c", 0]], dtype ="str")
        E12g = np.array([[0, 0, "-c"], [0, 0, 0], ["-c", 0, 0]], dtype ="str")
        E21g = np.array([["d", 0, 0], [0, "-d", 0], [0, 0, 0]], dtype ="str")
        E22g = np.array([[0, "-d", 0], ["-d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A1g", A1g, "E1g", E11g, "E1g", E12g, "E2g", E21g, "E2g", E22g]

    elif pointgroup == "23":
        A = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype ="str")
        E1 = np.array([["b+\u221A3c", 0, 0], [0, "b-\u221A3c", 0], [0, 0, "-2b"]], dtype ="str")
        E2 = np.array([["c-\u221A3b", 0, 0], [0, "c+\u221A3b", 0], [0, 0, 0]], dtype ="str")
        Tx = np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype ="str")
        Ty = np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype ="str")
        Tz = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A", A, "1E", E1, "2E", E2, "T(x)", Tx, "T(y)", Ty, "T(z)", Tz]

    elif pointgroup == "m-3":
        Ag = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype ="str")
        E1g = np.array([["b+\u221A3c", 0, 0], [0, "b-\u221A3c", 0], [0, 0, "-2b"]], dtype ="str")
        E2g = np.array([["c-\u221A3b", 0, 0], [0, "c+\u221A3b", 0], [0, 0, 0]], dtype ="str")
        Tg1 = np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype ="str")
        Tg2 = np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype ="str")
        Tg3 = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["Ag", Ag, "1Eg", E1g, "2Eg", E2g, "Tg", Tg1, "Tg", Tg2, "Tg", Tg3]

    elif pointgroup == "432":
        Ag = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype ="str")
        E1 = np.array([["b", 0, 0], [0, "b", 0], [0, 0, "-2b"]], dtype ="str")
        E2 = np.array([["-\u221A3b", 0, 0], [0, "\u221A3b", 0], [0, 0, 0]], dtype ="str")
        T21 = np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype ="str")
        T22 = np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype ="str")
        T23 = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A1", A1, "E", E1, "E", E2, "T2", T21, "T2", T22, "T2", T23]

    elif pointgroup == "-43m":
        Ag = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype ="str")
        E1 = np.array([["b", 0, 0], [0, "b", 0], [0, 0, "-2b"]], dtype ="str")
        E2 = np.array([["-\u221A3b", 0, 0], [0, "\u221A3b", 0], [0, 0, 0]], dtype ="str")
        T2x = np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype ="str")
        T2y = np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype ="str")
        T2z = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A1", A1, "E", E1, "E", E2, "T2(x)", T2x, "T2(y)", T2y, "T2(z)", T2z]

    elif pointgroup == "m-3m":
        A1g = np.array([["a", 0, 0], [0, "a", 0], [0, 0, "a"]], dtype ="str")
        Eg1 = np.array([["b", 0, 0], [0, "b", 0], [0, 0, "-2b"]], dtype ="str")
        Eg2 = np.array([["-\u221A3b", 0, 0], [0, "\u221A3b", 0], [0, 0, 0]], dtype ="str")
        T2g1 = np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype ="str")
        T2g2 = np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype ="str")
        T2g3 = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype ="str")
        RamanTensors = ["A1g", A1g, "Eg", Eg1, "Eg", Eg2, "T2g", T2g1, "T2g", T2g2, "T2g", T2g3]
    
    # just in case...
    else:
        print("[RamanLib]: no valid point group found, exiting...")
        sys.exit(1)
    #

    return RamanTensors
#

def RamanSelectionRules(pointgroup, RTs):
    if pointgroup == "3m":
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
                           "A1(LO+TO) + E(LO+TO)",
                           "E(TO)",
                           "E(TO)",
                           "E(LO+TO)"]
        rightComponents = [RTs[3][rDir[0]]+RTs[5][rDir[0]],\
                           RTs[3][rDir[1]]+RTs[5][rDir[1]],\
                           RTs[3][rDir[2]]+RTs[5][rDir[2]],\
                           RTs[1][rDir[3]],\
                           RTs[1][rDir[4]]+RTs[3][rDir[4]]+RTs[5][rDir[4]],\
                           RTs[3][rDir[5]]+RTs[5][rDir[5]],\
                           RTs[3][rDir[6]]+RTs[5][rDir[6]],\
                           RTs[3][rDir[7]]+RTs[5][rDir[7]]]
    #
    return backscattering, backComponents, rightscattering, rightComponents

"""
def get_chi2():
    xx = np.genfromtxt("oxx", dtype=float)
    xy = np.genfromtxt("oxy", dtype=float)
    xz = np.genfromtxt("oxz", dtype=float)
    yy = np.genfromtxt("oyy", dtype=float)
    yz = np.genfromtxt("oyz", dtype=float)
    zz = np.genfromtxt("ozz", dtype=float)
    return [xx, yy, zz, xy, yz, xz]

def get_LO_correction(chi2_tmp, born, eps_inf, qdir, vol, eigvec, w):
    lenw = len(w)
    chi2 = np.empty((3,3,3,lenw), dtype=complex)
    
    chi2[0,0,0] = np.interp(w, [x[0] for x in chi2_tmp[0]], [complex(x[2], x[1]) for x in chi2_tmp[0]])
    chi2[0,0,1] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[2], x[1]) for x in chi2_tmp[3]])
    chi2[0,0,2] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[2], x[1]) for x in chi2_tmp[5]])
    chi2[1,0,0] = np.interp(w, [x[0] for x in chi2_tmp[0]], [complex(x[4], x[3]) for x in chi2_tmp[0]])
    chi2[1,0,1] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[4], x[3]) for x in chi2_tmp[3]])
    chi2[1,0,2] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[4], x[3]) for x in chi2_tmp[5]])
    chi2[2,0,0] = np.interp(w, [x[0] for x in chi2_tmp[0]], [complex(x[6], x[5]) for x in chi2_tmp[0]])
    chi2[2,0,1] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[6], x[5]) for x in chi2_tmp[3]])
    chi2[2,0,2] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[6], x[5]) for x in chi2_tmp[5]])

    chi2[0,1,0] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[2], x[1]) for x in chi2_tmp[3]])
    chi2[0,1,1] = np.interp(w, [x[0] for x in chi2_tmp[1]], [complex(x[2], x[1]) for x in chi2_tmp[1]])
    chi2[0,1,2] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[2], x[1]) for x in chi2_tmp[4]])
    chi2[1,1,0] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[4], x[3]) for x in chi2_tmp[3]])
    chi2[1,1,1] = np.interp(w, [x[0] for x in chi2_tmp[1]], [complex(x[4], x[3]) for x in chi2_tmp[1]])
    chi2[1,1,2] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[4], x[3]) for x in chi2_tmp[4]])
    chi2[2,1,0] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[6], x[5]) for x in chi2_tmp[3]])
    chi2[2,1,1] = np.interp(w, [x[0] for x in chi2_tmp[1]], [complex(x[6], x[5]) for x in chi2_tmp[1]])
    chi2[2,1,2] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[6], x[5]) for x in chi2_tmp[4]])

    chi2[0,2,0] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[2], x[1]) for x in chi2_tmp[5]])
    chi2[0,2,1] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[2], x[1]) for x in chi2_tmp[4]])
    chi2[0,2,2] = np.interp(w, [x[0] for x in chi2_tmp[2]], [complex(x[2], x[1]) for x in chi2_tmp[2]])
    chi2[1,2,0] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[4], x[3]) for x in chi2_tmp[5]])
    chi2[1,2,1] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[4], x[3]) for x in chi2_tmp[4]])
    chi2[1,2,2] = np.interp(w, [x[0] for x in chi2_tmp[2]], [complex(x[4], x[3]) for x in chi2_tmp[2]])
    chi2[2,2,0] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[6], x[5]) for x in chi2_tmp[5]])
    chi2[2,2,1] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[6], x[5]) for x in chi2_tmp[4]])
    chi2[2,2,2] = np.interp(w, [x[0] for x in chi2_tmp[2]], [complex(x[6], x[5]) for x in chi2_tmp[2]])

    #
    # formats
    # chi2[i][j][l][w]
    # eigvec[atom][k]
    # born[atom][l][k]
    # qdir[l]
    # eps_inf[l][k]
    corr = np.empty((3,3,lenw), dtype=complex)
    for w in range(lenw):
        for atom in range(nat):
            for k in range(3):
                tmp1 = 0
                tmp2 = 0
                tmp3 = 0
                for l in range(3):
                    tmp1 += qdir[l]*born[atom,l,k]
                    tmp3 += chi2[:,:,l,w]*qdir[l]
                    for ls in range(3):
                        tmp2 += qdir[l]*eps_inf[l,ls]*qdir[ls]
                    #
                #               angst^-3        e        yambo    angst?
                corr[:,:,w] += 8*np.pi/vol * tmp1/tmp2 * tmp3 * eigvec[atom][k]
            #
        #
    #                e->As               angst -> m   yambo -> pm/V     pm -> m       eps0
    corr = corr * 1.60217663+10**(-19) * 10**(-20) / 2.38721*10**(-9) * 10**(-12) / 8.8541878176*10**(-12)# unit: 1/m 
    return np.array([corr[0,0,:], corr[1,1,:], corr[2,2,:], corr[0,1,:], corr[1,2,:], corr[0,2,:]])
#
"""

