#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import sys
import os.path
from math import sqrt
import numpy as np

sys.dont_write_bytecode = True

"""
def rotate_alpha(file, matrix):
    # rotate the Raman Intensity by matrix
    data = np.genfromtxt(file)
    w = [x[0] for x in data]
    x11 = [x[1] for x in data]
    x22 = [x[2] for x in data]
    x33 = [x[3] for x in data]
    x12 = [x[4] for x in data]
    x13 = [x[6] for x in data]
    x23 = [x[5] for x in data]

    orig = np.array([[x11, x12, x13],[x12, x22, x23], [x13, x23, x33]])
    new = np.empty_like(orig)
    for j in range(len(x11)):
        new[:,:,j] = np.dot(matrix.transpose(), np.dot(orig[:,:,j], matrix))
    #
    # write to file
    f = open(file+"_rotated",'w')
    f.write("# Raman intensity at ?eV laser-wavelength\n")
    f.write("# freq/cm-1        xx         yy          zz        xy        yz        xz       avg\n")
    tmp = np.empty_like(data)
    for j in range(len(x11)):
        tmp[j,0] = w[j]
        tmp[j,1] = new[0,0,j]
        tmp[j,2] = new[1,1,j]
        tmp[j,3] = new[2,2,j]
        tmp[j,4] = new[0,1,j]
        tmp[j,5] = new[1,2,j]
        tmp[j,6] = new[0,2,j]
    #
    np.savetxt(f, tmp)
    f.close()
"""

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

