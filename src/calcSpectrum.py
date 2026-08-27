#!/usr/bin/env python

#
# Function for calculating the Raman spectrum for given Raman tensors
#
import numpy as np
from src.Symmetries import eps0, c_cm, h, kb, ev2rcm

def Lorentz(hw, ab, smear):
    fmax = np.max(hw)
    erange = np.arange(0, 1.1*fmax, np.max(smear)/10)
    spectrum = 0.0 * erange
    for i in range(len(hw)):
        spectrum +=  ab[i] * smear[i]  / ( (hw[i]-erange)**2 + smear[i]**2 )
    #
    return erange, spectrum
#

def broadenData(intensity, eigvals, modelist, smear):
    # apply smearing to Raman tensors from "getConstantRaman"
    
    w, Spectrum = Lorentz([eigvals[mode] for mode in modelist], [intensity[mode] for mode in modelist], [smear[mode] for mode in modelist])

    return np.array([w, Spectrum])
#

def getRamanIntensity(modelist, raman, eigvals, w0, polarization, temp, stokes):
    intensity = {}
    for mode in modelist:
        cm1 = eigvals[mode]
        n  = (-np.exp(-h * cm1 * c_cm/(kb * temp))+1)**(-1)
        prefactor = h / (32 * np.pi**3 * (c_cm/100)**4 * eps0**2) * ( 2 * np.pi * c_cm )**3 * 10**(-30)
        # anti-stokes
        if stokes == False:
            intensity[mode] = ( prefactor*np.abs(raman[mode][polarization])**2 * (ev2rcm*w0 + cm1)**4 * (n-1)/cm1 )
        # Stokes
        else:
            intensity[mode] = ( prefactor*np.abs(raman[mode][polarization])**2 * (ev2rcm*w0 - cm1)**4 * n/cm1 )
        #
    #
    return intensity
#

def getConstantRaman(ramantensors, modelist, w0):
    Raman = {}
    for mode in modelist:
        w_list = np.real(ramantensors[mode][:,0])
        alpha = []

        for i in range(1, 9):
            alpha.append(np.abs(np.interp([w0], w_list, ramantensors[mode][:,i]))[0])
        #
        Raman[mode] = {}
        Raman[mode]["xx"] = alpha[0]
        Raman[mode]["yy"] = alpha[1]
        Raman[mode]["zz"] = alpha[2]
        Raman[mode]["xy"] = alpha[3]
        Raman[mode]["yz"] = alpha[4]
        Raman[mode]["xz"] = alpha[5]
        Raman[mode]["perp"] = alpha[6]
        Raman[mode]["back"] = alpha[7]
    #

    return Raman
#

def calcSpectrum(ramantensors, Ramanmodelist, eigvals, w0, temp, smear, stokes):
    print("[calcSpectrum]: Calculating Raman spectrum of modes "+str(Ramanmodelist))
    #print("[calcSpectrum]: Note: check e.g. https://www.cryst.ehu.es/cryst/polarizationselrules.html for selection rules")
    print("[calcSpectrum]: Laser frequency set to "+str(w0)+"eV")
    print("[calcSpectrum]: Temperature set to "+str(temp)+"K")
    raman = getConstantRaman(ramantensors, Ramanmodelist, w0)
    spectrum = {}
    intensity = {}
    for polarization in ["xx", "yy", "zz", "xy", "yz", "xz", "perp", "back"]:
        intensity[polarization] = getRamanIntensity(Ramanmodelist, raman, eigvals, w0, polarization, temp, stokes)
        spectrum[polarization] = broadenData(intensity[polarization], eigvals, Ramanmodelist, smear)
    #
    intensity["yx"] = intensity["xy"]
    intensity["zy"] = intensity["yz"]
    intensity["zx"] = intensity["xz"]
    spectrum["yx"] = spectrum["xy"]
    spectrum["zy"] = spectrum["yz"]
    spectrum["zx"] = spectrum["xz"]

    print("[calcSpectrum]: Done.")
    return raman, intensity, spectrum
#