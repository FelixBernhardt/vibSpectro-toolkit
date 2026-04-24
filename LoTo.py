#!/usr/bin/env python

#
# Lib to account for LO modes
#


import numpy as np
from parserPhonopy import parsePhonopy
from RamanLib import flatten, e_charge
    
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

def getLOFreqs(path, eigvecs, eigvals, qdir):
    # get the LO modes corresponding to the direction to be analyzed
    #<phonopy --readfc --sym-fc --writedm --qpoints="0 0 0" --nac --q-direction="0 0 1">

    eigvals_pt, eigvecs_pt, norms_pt, qpoint_pt, basis, nat, elements, cPos, masses = parsePhonopy(path, qdir)

    # match the TO to the LO modes
    LoToDict = LOTOassign(eigvecs, eigvecs_pt)
    
    # reorder the frequencies
    eigvalsLO = np.empty_like(eigvals)
    for j in range(3*nat):
        eigvalsLO[j] = eigvals_pt[LoToDict[j]]
    #

    return eigvals_pt
#

def getChi2():
    # from yambo o.xx, test case
    # unit cm/V, gaussian
    xx = np.genfromtxt("oxx", dtype=float)
    xy = np.genfromtxt("oxy", dtype=float)
    xz = np.genfromtxt("oxz", dtype=float)
    yy = np.genfromtxt("oyy", dtype=float)
    yz = np.genfromtxt("oyz", dtype=float)
    zz = np.genfromtxt("ozz", dtype=float)
    # return in m/V, SI
    return 4*np.pi/(3*10e4)*1e-2*[xx, yy, zz, xy, yz, xz]

def getLOCorrection(path, chi2_tmp, born, eps_inf, qdir, vol, w, nat):

    chi2 = np.empty((3,3,3), dtype=complex)
    
    chi2[0,0,0] = np.interp([w], [x[0] for x in chi2_tmp[0]], [complex(x[2], x[1]) for x in chi2_tmp[0]])
    chi2[0,0,1] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[2], x[1]) for x in chi2_tmp[3]])
    chi2[0,0,2] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[2], x[1]) for x in chi2_tmp[5]])
    chi2[1,0,0] = np.interp([w], [x[0] for x in chi2_tmp[0]], [complex(x[4], x[3]) for x in chi2_tmp[0]])
    chi2[1,0,1] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[4], x[3]) for x in chi2_tmp[3]])
    chi2[1,0,2] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[4], x[3]) for x in chi2_tmp[5]])
    chi2[2,0,0] = np.interp([w], [x[0] for x in chi2_tmp[0]], [complex(x[6], x[5]) for x in chi2_tmp[0]])
    chi2[2,0,1] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[6], x[5]) for x in chi2_tmp[3]])
    chi2[2,0,2] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[6], x[5]) for x in chi2_tmp[5]])

    chi2[0,1,0] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[2], x[1]) for x in chi2_tmp[3]])
    chi2[0,1,1] = np.interp([w], [x[0] for x in chi2_tmp[1]], [complex(x[2], x[1]) for x in chi2_tmp[1]])
    chi2[0,1,2] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[2], x[1]) for x in chi2_tmp[4]])
    chi2[1,1,0] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[4], x[3]) for x in chi2_tmp[3]])
    chi2[1,1,1] = np.interp([w], [x[0] for x in chi2_tmp[1]], [complex(x[4], x[3]) for x in chi2_tmp[1]])
    chi2[1,1,2] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[4], x[3]) for x in chi2_tmp[4]])
    chi2[2,1,0] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[6], x[5]) for x in chi2_tmp[3]])
    chi2[2,1,1] = np.interp([w], [x[0] for x in chi2_tmp[1]], [complex(x[6], x[5]) for x in chi2_tmp[1]])
    chi2[2,1,2] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[6], x[5]) for x in chi2_tmp[4]])

    chi2[0,2,0] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[2], x[1]) for x in chi2_tmp[5]])
    chi2[0,2,1] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[2], x[1]) for x in chi2_tmp[4]])
    chi2[0,2,2] = np.interp([w], [x[0] for x in chi2_tmp[2]], [complex(x[2], x[1]) for x in chi2_tmp[2]])
    chi2[1,2,0] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[4], x[3]) for x in chi2_tmp[5]])
    chi2[1,2,1] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[4], x[3]) for x in chi2_tmp[4]])
    chi2[1,2,2] = np.interp([w], [x[0] for x in chi2_tmp[2]], [complex(x[4], x[3]) for x in chi2_tmp[2]])
    chi2[2,2,0] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[6], x[5]) for x in chi2_tmp[5]])
    chi2[2,2,1] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[6], x[5]) for x in chi2_tmp[4]])
    chi2[2,2,2] = np.interp([w], [x[0] for x in chi2_tmp[2]], [complex(x[6], x[5]) for x in chi2_tmp[2]])

    #
    # formats
    # chi2[i][j][l]
    # born[atom][l][k]
    # qdir[l]
    # eps_inf[l][k]
    # formula from https://www.nature.com/articles/s41524-024-01236-3

    corr = np.empty(7, dtype=complex)
    dirdict = {0: (0,0), 1: (1,1), 2: (2,2), 3: (0,1), 4: (1,2), 5: (0,2)}
    for dir in range(6):
        tmpEps = 0
        tmpZ = 0
        tmpChi2 = 0
        for j in range(3):
            m, n = dirdict[dir]
            tmpChi2 += chi2[m, n, j]
            for k in range(3):
                tmpEps += qdir[j]*eps_inf[j,k]*qdir[k]
                for atom in range(nat):
                    tmpZ += qdir[j]*born[atom,j,k]
                #
            #
        #
        corr[dir] += 8*np.pi / vol * ( tmpZ * e_charge) / tmpEps * tmpChi2
    #
    
    # units are now 10e-30 C/Vm^2
    return corr
#