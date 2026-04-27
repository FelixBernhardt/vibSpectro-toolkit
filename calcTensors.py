#!/usr/bin/env python

#
# This lib calculates the Raman tensors
#

import os, re
import numpy as np
from RamanLib import eps0, placzeck_invs

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

def align_omega(w1, w2, Im1_tmp, Re1_tmp, Im2_tmp, Re2_tmp):
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

def calc_raman(path, mode, eigval, w, Im1, Re1, Im2, Re2, stepsize, basis):
    # get the derivative with respect to phonon-mode

    V0 = np.linalg.det(basis) # angst^3

    I = np.empty((6, len(w)), dtype=complex)
    outfile = path+"Ramantensors/alpha_"+str(mode)+".dat"
    f = open(outfile, "w")
    f.write("# Raman tensor in 10^(-30) Cm^2/V\n")
    f.write("# mode: " +str(mode)+"   phonon freq: "+str(eigval)+"\n")
    f.write("# omega(eV)    xx        yy        zz        xy        yz        xz      avg\n")
    for i in range(len(w)):
        for j in range(6):
            I[j][i] = (complex(Re1[j][i]-Re2[j][i], Im1[j][i]-Im2[j][i]))/( 2*stepsize*10**(-10) ) * eps0 * V0
        #
        perp, back = placzeck_invs(I, i)
        f.write("{:5.5f} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e}\n"\
                .format(w[i], I[0][i-1], I[1][i-1], I[2][i-1], I[3][i-1], I[4][i-1], I[5][i-1], perp, back))
    #
    f.close()
#

def calcDegenerates(path, modes, labels, ramantensors):
    # this might be mathematically impossible !!
    # get the corresponding ramantensors
    Rn = []
    label = labels[modes[0]-1]
    for i in range(int(len(ramantensors)/2)):
        if ramantensors[2*i] == label:
            Rn.append(ramantensors[2*i+1])
        #
    #

    Rb = []
    # decompose general tensors into their coefficients
    for letter in ["a", "b", "c", "d", "e", "f"]:
        Rnc = []
        for R in Rn:
            foundOne = False
            tmp_R = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    tmp = re.findall(letter, R[i,j])
                    if tmp != []:
                        foundOne = True
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
            #
        #
    #

    # orthonormalization, E holds the basis matrizes that the calculated Ramantensor decomposes into
    M = np.column_stack([R.reshape(-1) for R in Rb])
    U = np.linalg.svd( M, full_matrices=False)[0]
    E = []
    for i in range(len(U[0])):
        E.append(U[:, i].reshape(3,3))
    #

    # get the calculated, degenerate raman tensor
    data = np.genfromtxt(path+"Ramantensors/alpha_"+str(modes[0])+".dat", dtype=complex)
    with open(path+"Ramantensors/alpha_"+str(modes[0])+".dat") as f:
        f.readline()
        eigval = f.readline().split()[-1]
    #

    w = []
    I = []
    #for i in [100]:
    for i in range(len(data)):
        w.append(np.real(data[i,0]))
        Atest = np.zeros((3,3), dtype=complex)
        Atest[0,0] = data[i,1]
        Atest[1,1] = data[i,2]
        Atest[2,2] = data[i,3]
        Atest[0,1] = data[i,4]
        Atest[1,2] = data[i,5]
        Atest[0,2] = data[i,6]
        Atest[1,0] = Atest[0,1]
        Atest[2,1] = Atest[1,2]
        Atest[2,0] = Atest[0,2]

        
        # decompose calculated tensor into its general tensor components
        x = np.array( np.linalg.lstsq( np.column_stack([tmp.reshape(-1) for tmp in E]), Atest.reshape(-1), rcond=1.e-12)[0] )

        # 1. Normalize the first vector
        norm = np.linalg.norm(x)

        if norm < 1e-18:
            # pick an arbitrary unit vector as v1
            v1 = np.zeros_like(x)
            v1[0] = 1.0
        else:
            v1 = x / norm
        #

        # 2. Build an orthonormal basis with v1 as first vector
        # Start with random matrix and insert v1
        n = len(x)
        M = np.random.randn(n, n) + 1j * np.random.randn(n, n)
        M[0] = v1

        # QR gives orthonormal rows if we transpose
        Q = np.linalg.qr(M.T)[0]
        V = Q.T   # rows are orthonormal vectors in coefficient space

        # 3. Construct degenerate Raman tensors and reconstruct original tensor
        degenerates = np.zeros((len(Rn),6,1), dtype=complex)
        for j in range(len(Rn)):
            coeffs = V[j]  # vector of length n
            print(coeffs)
            R_degen = sum(coeffs[k] * E[k] for k in range(n))
            degenerates[j] = [[R_degen[0,0]], [R_degen[1,1]], [R_degen[2,2]], [R_degen[0,1]], [R_degen[1,2]], [R_degen[0,2]]]
        #

        I.append(degenerates)
    #
    I = np.array(I, dtype=complex)

    # write to file
    for j in range(0,len(modes)):
        outfile = path+"Ramantensors/alpha_"+str(modes[j])+"_degen.dat"
        f = open(outfile, "w")
        f.write("# Raman tensor in 10^(-30) Cm^2/V\n")
        f.write("# mode: " +str(modes[j])+"   phonon freq: "+str(eigval)+"\n")
        f.write("# omega(eV)    xx        yy        zz        xy        yz        xz      perp      back\n")

        for i in range(len(w)):
            perp, back = placzeck_invs( I[i][j], 1)
            f.write("{:5.5f} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3f} {:.3f}\n"\
                .format(w[i], I[i][j][0][0], I[i][j][1][0], I[i][j][2][0], I[i][j][3][0], I[i][j][4][0], I[i][j][5][0], perp, back))
            #
        #
        f.close()
    #
#

def calcTensors(path, modelist, program, eigvals, norms, basis, degenerates, labels, ramantensors, stepsize):
    disps = [-1, 1]

    print("[calcTensors]: Calculating Raman tensors of modes " + str(modelist))
    if os.path.isdir(path+"Ramantensors") == False:
        os.system("mkdir "+path+"Ramantensors")
    #
    if program == "VASP":
        from parserVASP import getOpticsVASP
        iteration = 0
        for mode in modelist:
            if len(modelist) > 10:
                printProgressBar(iteration, len(modelist)-1)
            #
            eigval = eigvals[mode-1]
            norm = norms[mode-1]
            w1, Im1, Re1 = getOpticsVASP(path+"displacements/mode"+str(mode)+"_"+str(disps[0])+"/vasprun.xml")
            w2, Im2, Re2 = getOpticsVASP(path+"displacements/mode"+str(mode)+"_"+str(disps[1])+"/vasprun.xml")

            #print("[calcTensors]: Calculating mode "+str(mode))
            w, Im1, Re1, Im2, Re2 = align_omega(w1, w2, Im1, Re1, Im2, Re2)
            calc_raman(path, mode, eigval, w, Im1, Re1, Im2, Re2, stepsize, basis)
            iteration += 1
        #
        print("[calcTensors]: Done.")
    elif program == "QE":
        from parserQE import getOpticsQE
        iteration = 0
        for mode in modelist:
            if len(modelist) > 10:
                printProgressBar(iteration, len(modelist)-1)
            #
            eigval = eigvals[mode-1]
            norm = norms[mode-1]
            w1, Im1, Re1 = getOpticsQE(path+"displacements/mode"+str(mode)+"_"+str(disps[0]))
            w2, Im2, Re2 = getOpticsQE(path+"displacements/mode"+str(mode)+"_"+str(disps[1]))

            #print("[calcTensors]: Calculating mode "+str(mode))
            w, Im1, Re1, Im2, Re2 = align_omega(w1, w2, Im1, Re1, Im2, Re2)
            calc_raman(path, mode, eigval, w, Im1, Re1, Im2, Re2, stepsize, basis)
            iteration += 1
        #
        print("[calcTensors]: Done.")
    else:
        print("[calcTensors]: Format not implemented, exiting...")
    #

 
    # calculate degenerate raman tensors
    if degenerates != []:
        print("[calcTensors]: Calculating degenerate tensors...")
        print(degenerates)
        for modes in degenerates:
            if modes[0] in modelist:
                calcDegenerates(path, modes, labels, ramantensors)
            #
        #
        print("[calcTensors]: Done.")
    #

#