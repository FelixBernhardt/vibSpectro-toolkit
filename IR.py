#!/usr/bin/env python

#
# calculate ionic contribution to dielectric function, i.e. IR spectrum
#

import numpy as np
from RamanLib import flatten, Lorentz, eps0, c_cm, e_charge, amu
import matplotlib.pyplot as plt
import matplotlib as mpl

def plotIRspectrum(file, path):
    # data
    epsi_data = []
    epsr_data = []
    dft_raw_data = np.loadtxt(file) # format: wavelength (cm-1) Intensity (Imag, Real)
    dict = {0: "x", 1: "y", 2: "z", 3: "avg"}
    w_data = [x[0] for x in dft_raw_data]
    epsi_data.append( [x[1] for x in dft_raw_data] )
    epsr_data.append( [x[2] for x in dft_raw_data] )
    epsi_data.append( [x[3] for x in dft_raw_data] )
    epsr_data.append( [x[4] for x in dft_raw_data] )
    epsi_data.append( [x[5] for x in dft_raw_data] )
    epsr_data.append( [x[6] for x in dft_raw_data] )
    epsi_data.append( [x[7] for x in dft_raw_data] )
    epsr_data.append( [x[8] for x in dft_raw_data] )


    # Fonts
    plt.rcParams.update({
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "lualatex",
    })
    mpl.use('pgf')
    
    size = 12
    mpl.rcParams['font.size'] = size
    mpl.rcParams['axes.titlesize'] = size
    mpl.rcParams['axes.labelsize'] = size
    mpl.rcParams['xtick.labelsize'] = size
    mpl.rcParams['ytick.labelsize'] = size
    mpl.rcParams['legend.fontsize'] = size
    mpl.rcParams['figure.titlesize'] = size

    for j in range(4):
        # plotting    
        fig_width = 5.511 # inch
        mpl.rcParams['figure.figsize'] = [fig_width, fig_width/2]
        fig, ((ax1, ax2)) = plt.subplots(1,2, layout="constrained")
        if j == 3:
            fig.suptitle("spatially averaged polarization")
        else:
            fig.suptitle("IR: E||"+dict[j]+" polarization")
        #
        ax1.set_xlim([w_data[0], w_data[-1]])
        ax1.plot(w_data, epsr_data[j], color="black", label="Real")
        ax1.axhline(ls="dashed")
        ax1.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax1.set_ylabel("Re($\\varepsilon$)")

        ax2.set_xlim([w_data[0], w_data[-1]])
        ax2.set_ylim([0, np.max(epsi_data[j])*1.1])
        ax2.plot(w_data, epsi_data[j], color="black", label="Imag")
        ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax2.set_ylabel("Im($\\varepsilon$)")
    
        plt.savefig(path+"IR_"+dict[j]+".pdf")
    #
    print("[plotIRSpectrum]: Done.") 
#

def calcIR(path, modelist_reduced, degenerates, silent, acoustics, eigvals, eigvecs, basis, nat, masses, born, smearing, plotFlag):
    # add degenerate and raman silent modes together
    modelist = []
    for mode in modelist_reduced:
        if mode not in modelist and mode not in acoustics:
            modelist.append(mode)
        #
    for mode in silent:
        if mode not in modelist and mode not in acoustics:
            modelist.append(mode)
        #
    for mode in flatten(degenerates):
        if mode not in modelist and mode not in acoustics:
            modelist.append(mode)
        #
    #
    modelist = np.sort(np.array(modelist))
    
    # calculate imaginary part of the dielectric function
    # formula from https://aip.scitation.org/doi/pdf/10.1063/1.466753
    # and https://application.wiley-vch.de/books/sample/3527405062_c01.pdf
    # VibrationalSpectroscopyinLifeScience.FriedrichSiebertandPeterHildebrandt Copyright82008WILEY-VCHVerlagGmbH&Co.KGaA,Weinheim ISBN:978-3-527-40506-0
    numModes = len(modelist)
    Sm = np.zeros((3,numModes))
    IR_Im = []
    V0 = np.linalg.det(basis) # angst^3

    counter = 0
    for mode in modelist:
        for alpha in range(3):
            sum = 0
            for atom in range(nat):
                for beta in range(3):
                    sum += born[atom][beta][alpha] * e_charge * np.real(eigvecs[mode-1][atom][beta]) / np.sqrt(masses[atom]*amu)
                #
            #
            Sm[alpha][counter] = sum**2 / (eps0 * V0*10**(-30))
        #
        counter += 1
    #

    # apply the smearing
    for alpha in range(3):
        w, tmp = Lorentz([eigvals[mode-1] for mode in modelist], Sm[alpha], smearing)
        IR_Im.append(tmp)
    #
    IR_Im = np.array(IR_Im)/( 2 * np.pi * c_cm )

    # calculate real part using Kramers-Kronig
    IR_Re = np.zeros((3,len(w)))
    for dir in range(3):
        for freq in range(len(w)):
            counter = 0
            for mode in modelist:
                IR_Re[dir][freq] += Sm[dir][counter] * ( eigvals[mode-1]**2 - w[freq]**2 ) / ( (eigvals[mode-1]**2 - w[freq]**2)**2 + smearing**2 * w[freq]**2 )
                counter += 1
            #
        #
    #
    IR_Re = IR_Re/( 2 * np.pi * c_cm )**2

    # write dielectric function to file
    output_fh = open(path+"IR.dat", "w")
    output_fh.write("# freq(cm-1)   E||x             E||y            E||z            avg\n")
    output_fh.write("#            Im    Re         Im    Re        Im    Re        Im    Re\n")
    for i in range(len(w)):
        output_fh.write("{:4.3f}     {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}\n".format(\
            w[i], IR_Im[0][i], IR_Re[0][i], IR_Im[1][i], IR_Re[1][i], IR_Im[2][i], IR_Re[2][i], 
            1/3*(IR_Im[0][i]+IR_Im[1][i]+IR_Im[2][i]), 1/3*(IR_Re[0][i]+IR_Re[1][i]+IR_Re[2][i])))
    #
    output_fh.close()
    
    print("[calcIR]: DONE")

    if plotFlag == True:
        plotIRspectrum(path+"IR.dat", path)
    #
#
