#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import os
import numpy as np
from RamanLib import flatten, portoq, eps0, c_cm, h, kb, ev2rcm
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

def broadenData(raman, w0, col, temp, smear, stokes):
    # apply smearing to Raman tensors from "writeConstantRaman"
    
    cm1 = np.real(raman[:,0])
    # calculate the Raman intensity for each mode and component
    n  = (-np.exp(-h * cm1 * c_cm/(kb * temp))+1)**(-1)
    prefactor = h / (32 * np.pi**3 * (c_cm/100)**4 * eps0**2) * ( 2 * np.pi * c_cm )**3 * 10**(-30)

    # anti-stokes
    if stokes == False:
        intensity = np.abs(raman[:,col+1])**2 * (ev2rcm*w0 + cm1)**4 * (n-1)/cm1
    # Stokes
    else:
        intensity = np.abs(raman[:,col+1])**2 * (ev2rcm*w0 - cm1)**4 * n/cm1
    #

    w, Spectrum = Lorentz(cm1, intensity, smear)

    return np.array([w, prefactor*Spectrum])
#

def getConstantRaman(path, filelist, modelist, eigvals, eigvecs, w0, basis, nat, born, eps_inf, qdir, LOcorr):
    # apply LO correction if needed
    if LOcorr == True:
        eigvalsLO_all = getLOFreqs(path, eigvecs, eigvals, qdir)
        eigvalsLO = np.empty(len(modelist))
        counter = 0
        for mode in modelist:
            eigvalsLO[counter] = eigvalsLO_all[mode-1]
            counter += 1
        #
        V0 = np.linalg.det(basis)
        LOTerm = getLOCorrection(born, eps_inf, qdir, V0, w0, nat, eigvecs)
    else:
        LOTerm = np.zeros(8)
    #
    
    # read "alpha_X.dat" and collect raman-shift at laser-wavelength w0
    Raman = []
    eigvals = []
    for file in filelist:
        data = np.genfromtxt(file, dtype=complex)
        with open(file) as f:
            f.readline()
            eigval = f.readline().split()[-1]
        #
        eigvals.append(eigval) 
        w_list = np.real(data[:,0])
        alpha = []

        for i in range(1, 9):
            alpha.append(np.abs(np.interp([w0], w_list, data[:,i]))[0] + LOTerm[i-1])
        #
        Raman.append(alpha)
    #

    if LOcorr == True:
        eigvals = np.real(eigvalsLO)
    #
    tmp = np.array(Raman)
    raman = np.insert(tmp, 0, eigvals, axis=1)

    return np.array(raman)
#

def writeConstantRaman(path, raman, w0, qdir):
    print("[writeConstantRaman]: Writing Raman_"+portoq[qdir]+"_"+str(w0)+"eV.dat")
    f = open(path+"Raman_"+portoq[qdir]+"_"+str(w0)+"eV.dat",'w')
    f.write("# Raman tensors (10^(-30) Cm^2/V) at "+str(w0)+"eV laser-wavelength and q-direction "+str(qdir)+"\n")
    f.write("# freq/cm-1        xx         yy          zz        xy        yz        xz        perp        back\n")
    np.savetxt(f, raman, fmt='%4.8f')
    f.close()
    print("[writeConstantRaman]: Done.")
#

def writeSpectrum(path, w0, spectrum):
    print("[writeSpectrum]: Writing Raman spectrum.")
    f = open(path+"Intensity_"+str(w0)+"eV.dat",'w')
    f.write("# Raman intensity at "+str(w0)+"eV laser-wavelength\n")
    f.write("# freq/cm-1        xx         yy          zz        xy        yz        xz       perp       back\n")
    for i in range(len(spectrum[0][0])):
        f.write("{:5.5f} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e}\n"\
        .format(spectrum[0][0][i], spectrum[0][1][i], spectrum[1][1][i], spectrum[2][1][i], spectrum[3][1][i], spectrum[4][1][i], spectrum[5][1][i], spectrum[6][1][i], spectrum[7][1][i]))
    f.close()
    print("[writeSpectrum]: Done.")
#


def calcSpectrum(path, modelist_reduced, degenerates, acoustics, eigvals, eigvecs, basis, nat, born, eps_inf, w0, temp, smear, stokes, qdir, LOcorr):
    # add the degenerate modes back in
    modelist = []
    for mode in modelist_reduced:
        if mode not in modelist and mode not in acoustics:
            modelist.append(mode)
        #
    for mode in flatten(degenerates):
        if mode not in modelist and mode not in acoustics:
            modelist.append(mode)
        #
    #
    modelist = np.sort(np.array(modelist))
    filelist = []
    for mode in modelist:
        filelist.append(path+"Ramantensors/alpha_"+str(mode)+".dat")

    print("[calcSpectrum]: Calculating Raman spectrum of modes "+str(modelist))
    #print("[calcSpectrum]: Note: check e.g. https://www.cryst.ehu.es/cryst/polarizationselrules.html for selection rules")
    print("[calcSpectrum]: Laser frequency set to "+str(w0)+"eV")
    print("[calcSpectrum]: Temperature set to "+str(temp)+"K")
    print("[calcSpectrum]: Smearing width set to "+str(smear)+"cm^-1")
    raman = getConstantRaman(path, filelist, modelist, eigvals, eigvecs, w0, basis, nat, born, eps_inf, qdir, LOcorr)
    spectrum = []
    for col in range(8):
        spectrum.append(broadenData(raman, w0, col, temp, smear, stokes))
    #
    print("[calcSpectrum]: Done.")
    return raman, np.array(spectrum)
#