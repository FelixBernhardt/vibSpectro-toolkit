#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import sys
import os.path
import numpy as np

def file_check(modelist, type):
    check = True
    for mode in modelist:
        if type == "vasprun":
            for j in ["_-1", "_1"]:
                if os.path.isfile("vasprun_"+str(mode)+j):
                    continue
                else:
                    print("[file_check]: missing file vasprun_"+str(mode)+j+", exiting...")
                    check = False
                #
            #
        elif type == "alpha":
            if os.path.isfile("alpha_"+str(mode)+".dat"):
                continue
            else:
                print("[file_check]: missing file alpha_"+str(mode)+".dat, exiting...")
                check = False
            #
        else:
            print("[file_check]: something went terribly wrong, exiting...")
            check = False
        #
    #
    if check == False:
        sys.exit(1)
    #
#

def to_plot(hw,ab,gam=0.001):
    fmin = min(hw)
    fmax = max(hw)
    erange = np.arange(fmin-40*gam,fmax+40*gam,gam/10)
    spectrum = 0.0*erange
    for i in range(len(hw)):
        spectrum += ab[i]*1/np.pi*gam/((hw[i]-erange)**2+gam**2)
    #
    return erange, spectrum
#

def broaden_data(datafile, w0, col, temp, smear):
    # apply smearing to Raman-tensors from "write_raman"
    c = 299792458            # m/s
    h = 6.62606957*10**(-34) # Js
    kb = 1.3806488*10**(-23) # J/K
    dict = {0: 'xx', 1: 'yy', 2: 'zz', 3: 'xy', 4: 'yz', 5: 'xz', 6: 'avg'}
    # check for imaginary modes and dont read them
    tmp = np.genfromtxt(datafile, dtype=float)
    imag_counter = 0
    for j in reversed(tmp[:,0]):
        if j > 0:
            break
        imag_counter += 1
    #
    if imag_counter > 0:
        print("[broaden_data]: Ignoring modes with imaginary frequency!")
    #
    hw = np.genfromtxt(datafile, skip_footer=imag_counter, dtype=float)
    cm1 = hw[:,0]
    # calculate the Raman intensity for each mode and component
    n  = (-np.exp(-h * cm1 * c * 100/(kb * temp))+1)**(-1)
    int1 = np.abs(hw[:,col+1])**2 * (8065.5401*w0 - cm1)**4 * n/cm1
    int1 /= np.max(np.abs(int1), axis=0)
    Es1, Spectrum1 = to_plot(cm1, int1, smear)
    filename = 'Intensity_'+str(dict[col])+".dat"
    f = open(filename,'w')
    f.write('# freq/cm-1  Intensity \n')
    for i in range(len(Es1)):
        f.write('%.5e   %.5e\n' % (Es1[i],Spectrum1[i]))
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
    f.write("# Raman tensors (absolute values) at "+str(w0)+"eV laser-wavelength\n")
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


def calcSpectrum(modelist, w0, temp, smear):
    print("[calcSpectrum]: Calculating Raman spectrum")
    print("[calcSpectrum]: Note: check e.g. https://www.cryst.ehu.es/cryst/polarizationselrules.html for selection rules")
    print("[calcSpectrum]: Laser frequency set to "+str(w0)+"eV")
    print("[calcSpectrum]: Temperature set to "+str(temp)+"K")
    print("[calcSpectrum]: Smearing width set to "+str(smear)+"cm^-1")

    filelist = []

    file_check(modelist, "alpha")
    for mode in modelist:
        filelist.append("alpha_"+str(mode)+".dat")
    #
    # write Raman tensor for all modes at laser-wavelength w0
    print("[__main__]: Writing Raman_"+str(w0)+"eV.dat")
    write_raman(filelist, w0)
    #
    print("[__main__]: Broadening spectrum")
    for col in range(7):
        broaden_data("Raman_"+str(w0)+"eV.dat", w0, col, temp, smear)
    #
    cat_broaden(w0)
    print("[__main__]: Done.")
    sys.exit(1)
#