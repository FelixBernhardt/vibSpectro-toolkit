#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import os
import numpy as np
from Symmetries import flatten, portoq, eps0, c_cm, h, kb, ev2rcm
from LoTo import getLOFreqs, getLOCorrection

def Lorentz(hw, ab, gam=0.001):
    fmax = max(hw)
    erange = np.arange(0, 1.1*fmax, gam/10)
    spectrum = 0.0 * erange
    for i in range(len(hw)):
        spectrum +=  ab[i] * gam  / ( (hw[i]-erange)**2 + gam**2 )
    #
    return erange, spectrum
#

def broadenData(modelist, raman, eigvals, w0, col, temp, smear, stokes):
    # apply smearing to Raman tensors from "writeConstantRaman"
    
    # calculate the Raman intensity for each mode and component
    intensity = []
    for mode in modelist:
        cm1 = eigvals[mode]
        n  = (-np.exp(-h * cm1 * c_cm/(kb * temp))+1)**(-1)
        prefactor = h / (32 * np.pi**3 * (c_cm/100)**4 * eps0**2) * ( 2 * np.pi * c_cm )**3 * 10**(-30)
        # anti-stokes
        if stokes == False:
            intensity.append( prefactor*np.abs(raman[mode][col])**2 * (ev2rcm*w0 + cm1)**4 * (n-1)/cm1 )
        # Stokes
        else:
            intensity.append( prefactor*np.abs(raman[mode][col])**2 * (ev2rcm*w0 - cm1)**4 * n/cm1 )
        #
    #

    w, Spectrum = Lorentz([eigvals[mode] for mode in modelist], intensity, smear)

    return np.array([w, Spectrum])
#

def getConstantRaman(path, ramantensors, modelist, eigvals, eigvecs, w0, basis, nat, born, eps_inf, qdir, LOcorr):
    # apply LO correction if needed
    if LOcorr == True:
        V0 = np.linalg.det(basis)
        LOTerm = getLOCorrection(path, born, eps_inf, qdir, V0, w0, nat, eigvecs)
    else:
        LOTerm = np.zeros(8)
    #
    
    # read ramantensors and collect raman-shift at laser-wavelength w0
    Raman = {}
    for mode in modelist:
        w_list = np.real(ramantensors[mode][:,0])
        alpha = []

        for i in range(1, 9):
            alpha.append(np.abs(np.interp([w0], w_list, ramantensors[mode][:,i]))[0] + LOTerm[i-1])
        #
        Raman[mode] = alpha
    #

    return Raman
#

def calcSpectrum(path, ramantensors, modelist, Ramanmodelist, eigvals, eigvecs, basis, nat, born, eps_inf, w0, temp, smear, stokes, qdir, LOcorr):
    print("[calcSpectrum]: Calculating Raman spectrum of modes "+str(modelist))
    #print("[calcSpectrum]: Note: check e.g. https://www.cryst.ehu.es/cryst/polarizationselrules.html for selection rules")
    print("[calcSpectrum]: Laser frequency set to "+str(w0)+"eV")
    print("[calcSpectrum]: Temperature set to "+str(temp)+"K")
    print("[calcSpectrum]: Smearing width set to "+str(smear)+"cm^-1")
    raman = getConstantRaman(path, ramantensors, Ramanmodelist, eigvals, eigvecs, w0, basis, nat, born, eps_inf, qdir, LOcorr)
    spectrum = []
    for col in range(8):
        spectrum.append(broadenData(modelist, raman, eigvals, w0, col, temp, smear, stokes))
    #
    print("[calcSpectrum]: Done.")
    return raman, np.array(spectrum)
#