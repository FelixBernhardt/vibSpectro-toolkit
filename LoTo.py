#!/usr/bin/env python

#
# Lib to account for LO modes
#

import sys
import os.path
import numpy as np
from parserPhonopy import parsePhonopy

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


def LOTOassign(eigvecs1, eigvecs2):
    nat = int(len(eigvecs1)/3)
    # assign phonopy LO-TO splitting
    keys = []
    values = []
    assigned = []
    certain = []
    for i in range(3*nat):
        prod = []
        for j in range(3*nat):
            if j not in assigned:
                prod.append(np.abs(np.dot(flatten(eigvecs1[i]), flatten(eigvecs2[j]))))
            else:
                prod.append(0.0)
            #
        index = max(range(len(prod)), key=prod.__getitem__)
        #if prod[index] < 0.9:
        #    print(prod[index])
        #    print(prod)
        #    print(str(eigvals1[i]) + " -> " + str(eigvals2[index]))
        certain.append(prod[index])
        assigned.append(index)
        keys.append(i)
        values.append(index)
    #
    #print(certain)
    #print(np.mean(certain))
    #print(np.min(certain))
    # only one assignment is to be allowed with a certainty < 0.5
    counter = 0
    for j in certain:
        if j < 0.1:
            counter += 1
        #
    #
    #if counter > 2:
    #    print("ERROR: THERE IS A PROBLEM ASSIGNING THE MODES")
    #    sys.exit(1)
    #
    mode_dict = dict(zip(keys, values))
    return mode_dict
#

def getLOFreqs(modelist, eigvecs, eigvals, porto):
    # get the LO modes corresponding to the direction to be analyzed
    #<phonopy --readfc --sym-fc --writedm --qpoints="0 0 0" --nac --q-direction="0 0 1">

    phonopy_fh = open("qpoints_"+porto+".yaml", "r")
    eigvals_pt, eigvecs_pt, norms_pt, qpoint_pt, basis, nat, elements, cPos, masses = parsePhonopy(modelist, porto)
    phonopy_fh.close()

    # match the TO to the LO modes
    LoToDict = LOTOassign(eigvecs, eigvecs_pt)
    
    # reorder the frequencies
    eigvalsLO = np.empty_like(eigvals)
    for j in range(3*nat):
        eigvalsLO[j] = eigvals_pt[LoToDict[j]]
    #

    return eigvals_pt

def get_chi2():
    xx = np.genfromtxt("oxx", dtype=float)
    xy = np.genfromtxt("oxy", dtype=float)
    xz = np.genfromtxt("oxz", dtype=float)
    yy = np.genfromtxt("oyy", dtype=float)
    yz = np.genfromtxt("oyz", dtype=float)
    zz = np.genfromtxt("ozz", dtype=float)
    return [xx, yy, zz, xy, yz, xz]

def get_LO_correction(chi2_tmp, born, eps_inf, qdir, vol, eigvec, w, nat):
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