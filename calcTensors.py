#!/usr/bin/env python

#
# This lib calculates the Raman tensors
#

import sys
# smoothen dielectric function in "read_optics"
#from scipy.signal import savgol_filter
from RamanLib import *
from parserPhonopy import parsePhonopy

def align_omega(w1, w2, Im1_tmp, Re1_tmp, Im2_tmp, Re2_tmp):
    w = np.linspace(0, np.min([w1[-1], w2[-1]]), num=np.min([len(w1), len(w2)]))
    Im1 = [[],[],[],[],[],[]]
    Re1 = [[],[],[],[],[],[]]
    Im2 = [[],[],[],[],[],[]]
    Re2 = [[],[],[],[],[],[]]
    for j in range(6):
        Im1[j] = np.interp(w, w1, Im1_tmp[j])
        Re1[j] = np.interp(w, w1, Re1_tmp[j])
        Im2[j] = np.interp(w, w2, Im2_tmp[j])
        Re2[j] = np.interp(w, w2, Re2_tmp[j])
    #
    return w, Im1, Re1, Im2, Re2
#

def calc_raman(mode, eigval, w, Im1, Re1, Im2, Re2, stepsize):
    # get the derivative with respect to phonon-mode
    I = [[],[],[],[],[],[]]
    outfile = "alpha_"+str(mode)+".dat"
    f = open(outfile, "w")
    f.write("# Raman tensor\n")
    f.write("# mode: " +str(mode)+"   phonon freq: "+str(eigval)+"\n")
    f.write("# omega(eV)    xx        yy        zz        xy        yz        xz      avg\n")
    for i in range(1,len(w)-1):
        for j in range(6):
            I[j].append(complex(Re1[j][i]-Re2[j][i], Im1[j][i]-Im2[j][i]))/( 2*stepsize)
        #
        # get Placzeck-invariants
        G0 = np.abs(I[0][i-1] + I[1][i-1] + I[2][i-1])**2/3.0
        G1 = 0
        G2 = (np.abs(I[0][i-1] - I[1][i-1])**2 \
              + np.abs(I[0][i-1] - I[2][i-1])**2 \
              + np.abs(I[1][i-1] - I[2][i-1])**2)/3.0 \
              + 2*(np.abs(I[3][i-1])**2 + np.abs(I[4][i-1])**2 + np.abs(I[5][i-1])**2)
        avg = np.sqrt(10*G0 + 5*G1 + 7*G2) # parallel and perpendicular components added together
        f.write("{:5.5f} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e}\n"\
                .format(w[i], I[0][i-1], I[1][i-1], I[2][i-1], I[3][i-1], I[4][i-1], I[5][i-1], avg))
    #
    f.close()
#

def calcTensors(modeList, program, stepsize):
    # get phonon modes and unit cell
    phonopy_fh = open("qpoints.yaml", "r")
    eigvals, eigvecs, norms = parsePhonopy(phonopy_fh)
    phonopy_fh.close()

    print("[calcTensors]: Calculating Raman tensors...")
    
    if program == "VASP":
        from parserVASP import parseOptics
        #file_check(modeList, "vasprun", LO_dir)
        breakout = 0
        iteration = 0
        total = len(modeList)
        for mode in modeList:
            printProgressBar(iteration, total-1)
            eigval = eigvals[mode-1]
            eigvec = eigvecs[mode-1]
            norm = norms[mode-1]
            breakout = parseOptics("vasprun"+str(mode)+"_1", breakout)
            w1, Im1, Re1 = read_optics("optics.dat")
            breakout = parseOptics("vasprun"+str(mode)+"_-1", breakout)
            w2, Im2, Re2 = read_optics("optics.dat")
            if breakout == 0:
                #print("[calcTensors]: Calculating mode "+str(mode))
                w, Im1, Re1, Im2, Re2 = align_omega(w1, w2, Im1, Re1, Im2, Re2)
                calc_raman(mode, eigval, w, Im1, Re1, Im2, Re2, stepsize)
            #
            iteration += 1
        #
        os.system("rm grep_optics.sh")
        os.system("rm optics.dat")
        if breakout == 1:
            sys.exit(1)
        #
        print("[__main__]: Done.")
        sys.exit(1)
    #