#!/usr/bin/env python

#
# calculate ionic contribution to dielectric function, i.e. IR spectrum
#

import sys
import numpy as np
from parserPhonopy import parsePhonopy
from calcSpectrum import Lorentz
from RamanLib import removeModes
from LoTo import getLOFreqs
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib as mpl

def plotIRspectrum(file):
    # data
    epsi_data = []
    epsr_data = []
    dft_raw_data = np.loadtxt(file) # format: wavelength (cm-1) Intensity (Imag, Real)
    dict = {0: "x", 1: "y", 2: "z"}
    w_data = [x[0] for x in dft_raw_data]
    epsi_data.append( [x[1] for x in dft_raw_data] )
    epsr_data.append( [x[2] for x in dft_raw_data] )
    epsi_data.append( [x[3] for x in dft_raw_data] )
    epsr_data.append( [x[4] for x in dft_raw_data] )
    epsi_data.append( [x[5] for x in dft_raw_data] )
    epsr_data.append( [x[6] for x in dft_raw_data] )

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

    for j in range(3):
        # plotting    
        fig_width = 5.511 # inch
        mpl.rcParams['figure.figsize'] = [fig_width, fig_width/2]
        fig, ((ax1, ax2)) = plt.subplots(1,2, layout="constrained")
        fig.suptitle("IR: E||"+dict[j]+" polarization")

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
    
        plt.savefig("IR_"+dict[j]+".pdf")
    #
    print("[plotIRSpectrum]: Done.")
    sys.exit(1)    
#

def calcIR(modelist_orig, program, smearing):
    # get TO phonon modes at Gamma and the unit cell
    eigvals, eigvecs, norms, qpoint, basis, nat, elements, cPos, masses = parsePhonopy(modelist_orig, None)

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

    # calculate imaginary part of the dielectric function
    # formula from https://aip.scitation.org/doi/pdf/10.1063/1.466753
    # and https://application.wiley-vch.de/books/sample/3527405062_c01.pdf
    # VibrationalSpectroscopyinLifeScience.FriedrichSiebertandPeterHildebrandt Copyright82008WILEY-VCHVerlagGmbH&Co.KGaA,Weinheim ISBN:978-3-527-40506-0
    modelist = removeModes(eigvecs, eigvals, masses, modelist_orig)
    numModes = len(modelist)
    Sm = np.zeros((3,numModes))
    IR_Im = []
    V0 = np.linalg.det(basis) # angst^3

    e_charge = 1.602176634e-19 # C
    amu = 1.66053906660e-27 # kg
    eps0 = 8.8541878128e-12 # F/m
    c_cm = 2.99792458e10 # cm/s

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
    output_fh = open("IR.dat", "w")
    output_fh.write("# freq(cm-1)   E||x             E||y            E||z\n")
    output_fh.write("#            Im    Re         Im    Re        Im    Re\n")
    for i in range(len(w)):
        output_fh.write("{:4.3f}     {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}    {:+4.3f}  {:+4.3f}\n".format(\
            w[i], IR_Im[0][i], IR_Re[0][i], IR_Im[1][i], IR_Re[1][i], IR_Im[2][i], IR_Re[2][i]))
    #
    output_fh.close()

    plotIRspectrum("IR.dat")

    """
    # fitting procedure to get LO-zero crossings, needed ?
    porto = "zz"
    eigvalsLO_all = getLOFreqs(modelist_orig, eigvecs, eigvals, porto)
    gamma0 = 5


    eigvalsTO = np.empty(numModes)
    eigvalsLO = np.empty(numModes)
    counter = 0
    for mode in modelist:
        eigvalsTO[counter] = eigvals[mode-1]
        eigvalsLO[counter] = eigvalsLO_all[mode-1]
        counter += 1
    #
    """

    print("[calcIR]: DONE")
    sys.exit(1)
#
