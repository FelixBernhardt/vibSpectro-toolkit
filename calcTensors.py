#!/usr/bin/env python

#
# This lib calculates the Raman tensors
#

import os, re
import numpy as np
from Symmetries import eps0

# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
#

def placzeckInvs(Intensity, col):
    # get Placzeck-invariants
    G0 = np.abs(Intensity[0][col-1] + Intensity[1][col-1] + Intensity[2][col-1])**2/3.0
    G1 = 0
    G2 = (np.abs(Intensity[0][col-1] - Intensity[1][col-1])**2 \
          + np.abs(Intensity[0][col-1] - Intensity[2][col-1])**2 \
          + np.abs(Intensity[1][col-1] - Intensity[2][col-1])**2)/3.0 \
          + 2*(np.abs(Intensity[3][col-1])**2 + np.abs(Intensity[4][col-1])**2 + np.abs(Intensity[5][col-1])**2)
    perp = np.sqrt(5*G1 + 3*G2)
    back = np.sqrt(10*G0 + 4*G2)
    #avg = np.sqrt(10*G0 + 5*G1 + 7*G2) # parallel and perpendicular components added together
    return perp, back
#

def alignOmega(w1, w2, Im1_tmp, Re1_tmp, Im2_tmp, Re2_tmp):
    w = np.linspace(0, np.min([w1[-1], w2[-1]]), num=np.min([len(w1), len(w2)]))

    Im1 = np.empty((6, len(w)))
    Re1 = np.empty((6, len(w)))
    Im2 = np.empty((6, len(w)))
    Re2 = np.empty((6, len(w)))

    for j in range(6):
        Im1[j] = np.interp(w, w1, Im1_tmp[j])
        Re1[j] = np.interp(w, w1, Re1_tmp[j])
        Im2[j] = np.interp(w, w2, Im2_tmp[j])
        Re2[j] = np.interp(w, w2, Re2_tmp[j])
    #
    return w, Im1, Re1, Im2, Re2
#

def calcRaman(w, Im1, Re1, Im2, Re2, stepsize, basis):
    # get the derivative with respect to phonon-mode
    ramantensor = []
    V0 = np.linalg.det(basis) # angst^3

    I = np.empty((6, len(w)), dtype=complex)
    for i in range(len(w)):
        for j in range(6):
            I[j][i] = (complex(Re1[j][i]-Re2[j][i], Im1[j][i]-Im2[j][i]))/( 2*stepsize*10**(-10) ) * eps0 * V0
        #
        perp, back = placzeckInvs(I, i)
        ramantensor.append([w[i], I[0][i-1], I[1][i-1], I[2][i-1], I[3][i-1], I[4][i-1], I[5][i-1], perp, back])
    #
    return np.array(ramantensor)
#

def calcDegenerates(path, modes, labels, ramantensors, ramandata, eigval):
    # not used or tested !!
    # this might be mathematically impossible !!
    # get the corresponding ramantensors
    Rn = []
    indx = []
    label = labels[modes[0]]
    for i in range(int(len(ramantensors)/2)):
        if ramantensors[2*i] == label:
            indx.append(i)
        #        
        Rn.append(ramantensors[2*i+1])
        #
    #

    Rb = []
    Rb2 = []
    Rb3 = []
    # decompose general tensors into their coefficients
    for letter in ["a", "b", "c", "d", "e", "f"]:
        find = False
        Rnc = []
        tmp_R2 = np.zeros((3,3))
        for R in Rn:
            foundOne = False
            tmp_R = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    tmp = re.findall(letter, R[i,j])
                    if tmp != []:
                        foundOne = True
                        find = True
                        tmp = re.split(letter, R[i,j])
                    
                        if tmp[0] == "":
                            tmp_R[i,j] = 1
                        elif tmp[0] == "-":
                            tmp_R[i,j] = -1
                        elif len(tmp[0]) > 1:
                            if tmp[0][0] == "\u221A":
                                tmp_R[i,j] = np.sqrt(float(tmp[0][1]))
                            elif tmp[0][1] == "\u221A" and tmp[0][0] == "-":
                                tmp_R[i,j] = -np.sqrt(float(tmp[0][2]))
                            else:
                                tmp_R[i,j] = -float(tmp[0][-1])
                            #
                        else:
                            tmp_R[i,j] = float(tmp[0][-1])
                        #
                    #
                #
            #
        
            if foundOne == True:
                Rnc.append(tmp_R)
                Rb.append(np.array(tmp_R))
                Rb3.append([letter, np.array(tmp_R)])
                tmp_R2 = np.add(tmp_R2, Rb[-1])
            #
        #
        if find == True:
            Rb2.append(np.array(tmp_R2))
    #

    # reorder matrices by letters
    E = []
    for letter in ["a", "b", "c", "d", "e", "f"]:
        for j in range(len(Rb)):
            if letter == Rb3[j][0]:
                E.append(Rb3[j][1])
            #
        #
    #

    # get the calculated, degenerate raman tensor
    data = ramandata

    w = []
    I = []
    #for i in [10]:
    for i in range(len(data)):
        w.append(np.real(data[i,0]))
        A = np.zeros((3,3), dtype=complex)
        A[0,0] = data[i,1]
        A[1,1] = data[i,2]
        A[2,2] = data[i,3]
        A[0,1] = data[i,4]
        A[1,2] = data[i,5]
        A[0,2] = data[i,6]
        A[1,0] = A[0,1]
        A[2,1] = A[1,2]
        A[2,0] = A[0,2]
        A = np.nan_to_num(A, nan=0.0)

        #print(A)
        #for i in range(len(E)):
        #    print(E[i])

        # decompose calculated tensor into its general tensor components, all possible tensors and mode-mixing included
        x = np.real(np.array( np.linalg.lstsq( np.column_stack([tmp.reshape(-1) for tmp in E]), A.reshape(-1), rcond=1.e-5)[0] ))
        #print(x)

        # which tensors to ignore when constructing the degenerate tensors
        off = []
        for j in range(indx[0]):
            for k in range(3):
                for i in range(3):
                    if np.any(Rn[j][k,i] in off) or Rn[j][k,i] == "0":
                        continue
                    else:
                        off.append(Rn[j][k,i])
                    #
                #
            #
        #
        offset = len(off)
        lenmodes = len(modes)
        
        # construct Raman tensors in high symmetry form
        degenerates = np.zeros((lenmodes,9,1), dtype=complex)
        for j in range(lenmodes):
            R_degen = np.zeros((3,3))
            # get general Raman tensor components
            if lenmodes == 1:
                for k in range(2):
                    R_degen += np.sqrt(x[k+offset+1]**2+x[k+offset+2]**2)*E[j+2*k+offset]
            #
            elif lenmodes == 2:
                for k in range(2):
                    R_degen += np.sqrt(x[k+offset+1]**2+x[k+offset+2]**2)*E[j+2*k+offset]
                #
            elif lenmodes == 3:
                for k in range(3):
                    R_degen += np.sqrt(x[k+offset+1]**2+x[k+offset+2]**2+x[k+offset+3]**2)*E[j+3*k+offset]
                #
            #
            #print(R_degen)
            degenerates[j] = [[data[i,0]], [R_degen[0,0]], [R_degen[1,1]], [R_degen[2,2]], [R_degen[0,1]], [R_degen[1,2]], [R_degen[0,2]], [data[i,7]], [data[i,8]]]
        #
        I.append(degenerates)
    #
    I = np.swapaxes(np.array(I, dtype=complex), 1, 3)
    I = np.swapaxes(np.array(I, dtype=complex), 0, 1)

    return I[0,:,:,:]
#
        
def calcTensors(path, modelist, parser, eigvals, norms, basis, degenerates, labels, ramantensors, stepsize):
    disps = [-1, 1]
    ramantensor = {}

    print("[calcTensors]: Calculating Raman tensors of modes " + str(modelist))
    if os.path.isdir(path+"Ramantensors") == False:
        os.system("mkdir "+path+"Ramantensors")
    #
    for mode in modelist:
        eigval = eigvals[mode]
        w1, Im1, Re1 = parser.get_epsilon(path, mode , disps[0])
        w2, Im2, Re2 = parser.get_epsilon(path, mode , disps[1])

        w, Im1, Re1, Im2, Re2 = alignOmega(w1, w2, Im1, Re1, Im2, Re2)
        ramantensor[mode] = calcRaman(w, Im1, Re1, Im2, Re2, stepsize, basis)

        if mode in [x[0] for x in degenerates]:
            degeneratetensors = calcDegenerates(path, [mode, mode+1], labels, ramantensors, ramantensor[mode], eigval)
            counter = 0
            for degen in [mode, mode+1]:
                ramantensor[degen] = degeneratetensors[:,:,counter]
                counter += 1
        #
    #
    print("[calcTensors]: Done.")
    return ramantensor
#