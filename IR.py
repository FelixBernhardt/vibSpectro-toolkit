#!/usr/bin/env python

#
# calculate ionic contribution to dielectric function, i.e. IR spectrum
#

import sys
import numpy as np
from parserPhonopy import parsePhonopy
from calcSpectrum import to_plot
import matplotlib.pyplot as plt

def plotIRspectrum(porto):
    fontsize=12
    dft_raw_data = np.loadtxt("dielectric.dat") # format: wavelength (cm-1) Intensity (Imag, Real)
    dict = {"xx": 1, "yy": 3, "zz": 5, "xy": 7, "yz": 9, "xz": 11}
    x_data = [x[0] for x in dft_raw_data]
    yi_data = [x[dict[porto]] for x in dft_raw_data]
    yi_max = np.max(yi_data)

    yr_data = [x[dict[porto]+1] for x in dft_raw_data]
    yr_max = np.max(yr_data)

    ax = plt.subplot()
    ax.plot(x_data, yi_data/yi_max, color="black", label="Imag")
    ax.plot(x_data, yr_data/yr_max, color="red", label="Real")
    #ax.legend(fontsize=fontsize)
    ax.set_title("IR: "+str(porto)+" polarization")
    #ax.set_xlim(0,1000)
    #ax.set_ylim(0,1.1)
    #ax.set_yticks([])
    plt.xticks([0, 200, 400, 600, 800, 1000], labels=None, fontsize=fontsize)
    ax.set_xticklabels([0, 200, 400, 600, 800, 1000])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (arb. units)", fontsize=fontsize)
    plt.savefig("IR_"+str(porto)+".pdf")
    print("[plotIRSpectrum]: Done.")
    sys.exit(1)    
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


def phonopy_assign(eigvecs1, eigvecs2, nat):
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
        sum = 0
        for j in range(len(prod)):
            sum += np.abs(prod[j])**2
        #print(sum, np.abs(np.dot(flatten(eigvecs1[i]), flatten(eigvecs1[i])))**2 )
    #
    print(certain)
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


def calcIR(modelist, program, smearing, porto):
    # get TO phonon modes and unit cell
    phonopy_fh = open("qpoints.yaml", "r")
    eigvals, eigvecs, norms, qpoint, basis, nat, elements, cPos, masses = parsePhonopy(modelist, None)
    phonopy_fh.close()

    """
    # get the LO modes corresponding to the direction to be analyzed
    phonopy_fh = open("qpoints_"+porto+".yaml", "r")
    eigvals_pt, eigvecs_pt, norms_pt, qpoint_pt, basis, nat, elements, cPos, masses = parsePhonopy(modelist, porto)
    phonopy_fh.close()

    # match the TO to the LO modes
    LOTO = phonopy_assign(eigvecs, eigvecs, nat)
    print(LOTO)
    """

    # get BORN charges, in |e|
    if program == "VASP":
        from parserVASP import getBornVASP
        born = getBornVASP("OUTCAR", nat)
    elif program == "QE":
        from parserQE import getBornQE
        born = getBornQE("ph.out", nat)
    else:
        print("[calcIR]: Format not implemented, exiting..")
        sys.exit(1)
    #

    # ignore imaginary and acoustic modes
    modelist_tmp = []
    for mode in modelist:
        if eigvals[mode-1] > 10:
            modelist_tmp.append(mode)
        else:
            print("[calcIR]: Ignoring modes with imaginary frequency!")
        #
    modelist = modelist_tmp

    # calculate imaginary part of the dielectric function
    # formula from https://aip.scitation.org/doi/pdf/10.1063/1.466753
    # and https://application.wiley-vch.de/books/sample/3527405062_c01.pdf
    # VibrationalSpectroscopyinLifeScience.FriedrichSiebertandPeterHildebrandt Copyright82008WILEY-VCHVerlagGmbH&Co.KGaA,Weinheim ISBN:978-3-527-40506-0
    numModes = len(modelist)
    sum = np.zeros((numModes,3))
    S_xx = np.zeros((numModes))
    S_yy = np.zeros((numModes))
    S_zz = np.zeros((numModes))
    S_xy = np.zeros((numModes))
    S_yz = np.zeros((numModes))
    S_xz = np.zeros((numModes))
    V0 = np.linalg.det(basis) # angst^3
    prefactor = 4*np.pi**2 / ( 2 * V0 ) *1.5E8
    counter = 0
    for mode in modelist:
        for alpha in range(3):
            for atom in range(nat):
                for beta in range(3):
                    sum[counter][alpha] += born[atom][beta][alpha] * eigvecs[mode-1][atom][beta]
                #
            #
        #
        # Oscillator strength according to https://abinit14.sciencesconf.org/data/program/Lecture_May13_Rignanese_DFPT_Basics.pdf
        S_xx[counter] = np.abs(sum[counter][0])**2 * prefactor / eigvals[mode-1]
        S_yy[counter] = np.abs(sum[counter][1])**2 * prefactor / eigvals[mode-1]
        S_zz[counter] = np.abs(sum[counter][2])**2 * prefactor / eigvals[mode-1]
        S_xy[counter] = np.abs(sum[counter][0]*sum[counter][1]) * prefactor / eigvals[mode-1]
        S_yz[counter] = np.abs(sum[counter][1]*sum[counter][2]) * prefactor / eigvals[mode-1]
        S_xz[counter] = np.abs(sum[counter][0]*sum[counter][2]) * prefactor / eigvals[mode-1]
        counter += 1
    #

    # apply the smearing
    w, I_xx = to_plot([eigvals[mode-1] for mode in modelist], S_xx, smearing)
    dummy, I_yy = to_plot([eigvals[mode-1] for mode in modelist], S_yy, smearing)
    dummy, I_zz = to_plot([eigvals[mode-1] for mode in modelist], S_zz, smearing)
    dummy, I_xy = to_plot([eigvals[mode-1] for mode in modelist], S_xy, smearing)
    dummy, I_yz = to_plot([eigvals[mode-1] for mode in modelist], S_yz, smearing)
    dummy, I_xz = to_plot([eigvals[mode-1] for mode in modelist], S_xz, smearing)

    # calculate real part using Kramers-Kronig

    R_xx = np.zeros(len(w))
    R_yy = np.zeros(len(w))
    R_zz = np.zeros(len(w))
    R_xy = np.zeros(len(w))
    R_yz = np.zeros(len(w))
    R_xz = np.zeros(len(w))

    for freq in range(len(w)):
        counter = 0
        for mode in modelist:
            R_xx[freq] += S_xx[counter] * (eigvals[mode-1]**2-w[freq]**2) / ((eigvals[mode-1]**2-w[freq]**2)**2+smearing**2*freq**2)
            R_yy[freq] += S_yy[counter] * (eigvals[mode-1]**2-w[freq]**2) / ((eigvals[mode-1]**2-w[freq]**2)**2+smearing**2*freq**2)
            R_zz[freq] += S_zz[counter] * (eigvals[mode-1]**2-w[freq]**2) / ((eigvals[mode-1]**2-w[freq]**2)**2+smearing**2*freq**2)
            R_xy[freq] += S_xy[counter] * (eigvals[mode-1]**2-w[freq]**2) / ((eigvals[mode-1]**2-w[freq]**2)**2+smearing**2*freq**2)
            R_yz[freq] += S_yz[counter] * (eigvals[mode-1]**2-w[freq]**2) / ((eigvals[mode-1]**2-w[freq]**2)**2+smearing**2*freq**2)
            R_xz[freq] += S_xz[counter] * (eigvals[mode-1]**2-w[freq]**2) / ((eigvals[mode-1]**2-w[freq]**2)**2+smearing**2*freq**2)
            counter += 1
        #
    #

    # write dielectric function to file
    output_fh = open('dielectric.dat', 'w')
    output_fh.write("# freq(cm-1)  xx             yy               zz                xy               yz                  xz\n")
    output_fh.write("#        Im     Re        Im     Re        Im     Re         Im     Re        Im     Re           Im     Re\n")
    for i in range(len(w)):
        output_fh.write('{:4.3f}  {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}\n'.format(\
            w[i], I_xx[i], R_xx[i], I_yy[i], R_yy[i], I_zz[i], R_zz[i], I_xy[i], R_xy[i], I_yz[i], R_yz[i], I_xz[i], R_xz[i]))
    #
    output_fh.close()

    plotIRspectrum(porto)
    print("[calcIR]: DONE")
#
