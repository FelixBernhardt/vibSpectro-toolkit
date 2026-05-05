#!/usr/bin/env python

#
# calculate ionic contribution to dielectric function, i.e. IR spectrum
#

import numpy as np
from RamanLib import eps0, c_cm, e_charge, amu

def Lorentz_IR(hw, ab, gam=0.001):
    fmax = max(hw)
    erange = np.arange(0, 1.1*fmax, gam/10)
    spectrum = 0.0 * erange
    for i in range(len(hw)):
        spectrum +=  ab[i] * gam * hw[i]  / ( (hw[i]**2 - erange**2)**2 + erange**2 * gam**2 )
    #
    return erange, spectrum
#

def calcReflectance(path, file):
    data = np.genfromtxt(path+file)
    w = np.array([x[0] for x in data])
    col = []
    for j in range(1,8,2):
        epsi = np.array([x[j] for x in data])
        epsr = np.array([x[j+1] for x in data])
        tmp = []
        for i in range(len(w)):

            """
            omega   : array of angular frequencies [rad/s] (or use 2*pi*c / lambda)
            eps_ion : array of ionic dielectric contribution (scalar for chosen polarization)
            eps_inf : scalar high-frequency dielectric constant
            d       : thickness [cm]
            """

            eps = complex(epsr[i], epsi[i])          # total dielectric function
            n_complex = np.sqrt(eps)                 # n + i k

            # Fresnel reflectance at normal incidence (air–sample–air)
            r = (n_complex - 1) / (n_complex + 1)
            R = np.abs(r)**2

            tmp.append(R)
        #
        col.append(tmp)
    #

    # write dielectric function to file
    output_fh = open(path+"Reflectance.dat", "w")
    output_fh.write("# freq(cm-1)   E||x             E||y            E||z            avg\n")
    for i in range(len(w)):
        output_fh.write("{:4.3f}         {:+4.3f}           {:+4.3f}           {:+4.3f}           {:+4.3f}\n".format(\
            w[i], col[0][i], col[1][i], col[2][i], col[3][i]))
    #
    output_fh.close()
    
    print("[calcReflectance]: Done.")
#


def calcIR(path, modelist, eigvals, eigvecs, basis, nat, masses, born, smearing): 
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
            Sm[alpha][counter] = sum**2 / (eps0 * V0*10**(-30)) / ( 2 * np.pi * c_cm )**2
        #
        counter += 1
    #

    # apply the smearing
    for alpha in range(3):
        w, tmp = Lorentz_IR([eigvals[mode-1] for mode in modelist], Sm[alpha], smearing)
        IR_Im.append(tmp)
    #
    IR_Im = np.array(IR_Im)

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
    #IR_Re = IR_Re/( 2 * np.pi * c_cm )**2

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
    
    print("[calcIR]: Done.")
#
