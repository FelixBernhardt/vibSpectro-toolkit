#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import sys
import os.path
import numpy as np
from parserPhonopy import parsePhonopy
from RamanLib import removeModes

def Lorentz(hw, ab, gam=0.001):
    fmax = max(hw)
    erange = np.arange(0, 1.1*fmax, gam/10)
    spectrum = 0.0 * erange
    for i in range(len(hw)):
        spectrum += 1 / np.pi * ( ab[i] * gam ) / ( (hw[i]-erange)**2 + gam**2 )
    #
    return erange, spectrum
#

def broaden_data(datafile, w0, col, temp, smear):
    # apply smearing to Raman-tensors from "write_raman"
    c_cm = 2.99792458e10     # cm/s
    h = 6.62606957*10**(-34) # Js
    kb = 1.3806488*10**(-23) # J/K
    eps0 = 8.8541878128e-12  # F/m
    dict = {0: 'xx', 1: 'yy', 2: 'zz', 3: 'xy', 4: 'yz', 5: 'xz', 6: 'avg'}
    
    hw = np.genfromtxt(datafile, dtype=float)
    cm1 = hw[:,0]
    # calculate the Raman intensity for each mode and component
    n  = (-np.exp(-h * cm1 * c_cm/(kb * temp))+1)**(-1)
    prefactor = h / (32 * np.pi**3 * (c_cm/100)**4 * eps0**2) * ( 2 * np.pi * c_cm )**3 * 10**(-30)

    intensity = np.abs(hw[:,col+1])**2 * (8065.5401*w0 - cm1)**4 * n/cm1
    w, Spectrum = Lorentz(cm1, intensity, smear)
    filename = 'Intensity_'+str(dict[col])+".dat"
    f = open(filename,'w')
    f.write('# freq [cm-1]  Intensity [m^2/sr]\n')
    for i in range(len(w)):
        f.write('%.5e   %.5e\n' % (w[i], prefactor*Spectrum[i]))
    f.close()
#

def write_raman(filelist, w0):
    # read "alpha_X.dat" and collect raman-shift at laser-wavelenght w0
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
        index = 0
        alpha = []
        for w in w_list:
            # linear interpolation for each component
            if w > w0:
                for i in range(1,8):
                    # linear interpolation of the absolute values
                    y1 = np.abs(data[index-1,i])
                    y2 = np.abs(data[index,i])
                    x1 = w_list[index-1]
                    x2 = w_list[index]
                    m  = (y2-y1)/(x2-x1)
                    b  = y1-m*x1
                    
                    alpha.append(m*w0+b)
                #
                break
            #
            index += 1
        #
        Raman.append(alpha)
    #
    tmp = np.array(Raman)
    raman = np.insert(tmp, 0, eigvals, axis=1)
 
    f = open("Raman_"+str(w0)+"eV.dat",'w')
    f.write("# Raman tensors (10^(-30) Cm^2/V) at "+str(w0)+"eV laser-wavelength\n")
    f.write("# freq/cm-1        xx         yy          zz        xy        yz        xz        avg\n")
    np.savetxt(f, raman, fmt='%4.8f')
    f.close()
#

def cat_broaden(w0):
    # concat all broadened spectra into a single file
    filelist = []
    dict = {0: 'xx', 1: 'yy', 2: 'zz', 3: 'xy', 4: 'yz', 5: 'xz', 6: 'avg'}
    for col in range(7):
        filelist.append("Intensity_"+str(dict[col])+".dat")
    #
    data0 = np.genfromtxt(filelist[0], dtype=float)
    tmp = np.zeros((len(data0), 8))
    index = 0
    for file in filelist:
        data = np.genfromtxt(file, dtype=float)
        if index == 0:
            tmp[:,0] = data[:,0]
        #
        tmp[:,index+1] = data[:,1]
        index += 1
        os.system("rm "+file)
    #
    f = open("Intensity_"+str(w0)+"eV.dat",'w')
    f.write("# Raman intensity at "+str(w0)+"eV laser-wavelength\n")
    f.write("# freq/cm-1        xx         yy          zz        xy        yz        xz       avg\n")
    np.savetxt(f, tmp)
    f.close()
#


def calcSpectrum(modelist_orig, w0, temp, smear, porto):
    eigvals, eigvecs, norms, qpoint, basis, nat, elements, cPos, masses = parsePhonopy(modelist_orig, None)
    modelist = removeModes(eigvecs, eigvals, masses, modelist_orig)

    print("[calcSpectrum]: Calculating Raman spectrum of modes "+str(modelist))
    #print("[calcSpectrum]: Note: check e.g. https://www.cryst.ehu.es/cryst/polarizationselrules.html for selection rules")
    print("[calcSpectrum]: Laser frequency set to "+str(w0)+"eV")
    print("[calcSpectrum]: Temperature set to "+str(temp)+"K")
    print("[calcSpectrum]: Smearing width set to "+str(smear)+"cm^-1")

    filelist = []
    for mode in modelist:
        filelist.append("Ramantensors/alpha_"+str(mode)+".dat")
    #
    # write Raman tensor for all modes at laser-wavelength w0
    print("[calcSpectrum]: Writing Raman_"+str(w0)+"eV.dat")
    write_raman(filelist, w0)
    #
    print("[calcSpectrum]: Broadening spectrum")
    for col in range(7):
        broaden_data("Raman_"+str(w0)+"eV.dat", w0, col, temp, smear)
    #
    cat_broaden(w0)
    print("[calcSpectrum]: Done.")
    sys.exit(1)
#