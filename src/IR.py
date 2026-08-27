#!/usr/bin/env python

#
# calculate ionic contribution to dielectric function, i.e. IR spectrum
#

import numpy as np
from src.Symmetries import eps0, c_cm, e_charge, amu

def LorentzIR(hw, ab, smear):
    fmax = np.max(hw)
    erange = np.arange(0, 1.1*fmax, np.max(smear)/10)
    spectrum = 0.0 * erange
    for i in range(len(hw)):
        spectrum +=  ab[i] * smear[i] * hw[i]  / ( (hw[i]**2 - erange**2)**2 + erange**2 * smear[i]**2 )
    #
    return np.array(erange), np.array(spectrum)
#

def calcReflectance(IRdata):
    col = []
    for j in ["x", "y", "z"]:
        w = np.array(IRdata[j][0])
        epsi = np.imag(IRdata[j][1])
        epsr = np.real(IRdata[j][1])
        #
        tmp = []
        for i in range(len(w)):

            eps = complex(epsr[i], epsi[i])          # total dielectric function
            n_complex = np.sqrt(eps)                 # n + i k

            # Fresnel reflectance at normal incidence (air–sample–air)
            r = (n_complex - 1) / (n_complex + 1)
            R = np.abs(r)**2

            tmp.append(np.real(R))
        #
        col.append(tmp)
    #
    print("[calcReflectance]: Done.")
    rdata_dict = {}
    rdata_dict["x"] = [w, col[0]]
    rdata_dict["y"] = [w, col[1]]
    rdata_dict["z"] = [w, col[2]]
    return rdata_dict
#

def calcIR(modelist, eigvals, eigvecs, basis, nat, masses, born, smearing): 
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
                    sum += born[atom][beta][alpha] * e_charge * np.real(eigvecs[mode][atom][beta]) / np.sqrt(masses[atom]*amu)
                #
            #
            Sm[alpha][counter] = sum**2 / (eps0 * V0*10**(-30)) / ( 2 * np.pi * c_cm )**2
        #
        counter += 1
    #

    # apply the smearing
    for alpha in range(3):
        w, tmp = LorentzIR([eigvals[mode] for mode in modelist], Sm[alpha], [smearing[mode] for mode in modelist])
        IR_Im.append(tmp)
    #
    IR_Im = np.array(IR_Im)

    # calculate real part using Kramers-Kronig
    IR_Re = np.zeros((3,len(w)))
    for dir in range(3):
        for freq in range(len(w)):
            counter = 0
            for mode in modelist:
                IR_Re[dir][freq] += Sm[dir][counter] * ( eigvals[mode]**2 - w[freq]**2 ) / ( (eigvals[mode]**2 - w[freq]**2)**2 + smearing[mode]**2 * w[freq]**2 )
                counter += 1
            #
        #
    #

    # write output
    IRdata_dict = {}
    IRdata_dict["x"] = [w, [complex(IR_Re[0][i], IR_Im[0][i]) for i in range(len(w))]]
    IRdata_dict["y"] = [w, [complex(IR_Re[1][i], IR_Im[1][i]) for i in range(len(w))]]
    IRdata_dict["z"] = [w, [complex(IR_Re[2][i], IR_Im[2][i]) for i in range(len(w))]]

    print("[calcIR]: Done.")
    return IRdata_dict
#