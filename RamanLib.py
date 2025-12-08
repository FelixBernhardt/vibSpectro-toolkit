#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import re
import sys
import os.path
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
# smoothen dielectric function in "read_optics"
from scipy.signal import savgol_filter

sys.dont_write_bytecode = True

# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print("\n")
#

# read in dielectric function from grep_optics.sh
def read_optics(infile):
    with open(infile) as f:
        w = []
        Im = [[],[],[],[],[],[]]
        Re = [[],[],[],[],[],[]]

        for line in f:
            # first NEDOS values in imaginary part, leftovers in real
            if float(line.split()[0]) in w:
                for i in range(6):
                    if len(Re[i]) == len(Im[i]):
                        continue
                    else:
                        Re[i].append(float(line.split()[i+1]))
                    #
                #
            else:
                w.append(float(line.split()[0]))
                for i in range(6):
                    Im[i].append(float(line.split()[i+1]))
                #
            #
        #
    #
    # smoothen the data
    #Re_fit = savgol_filter(Re, 51, 5) # window size 51, polynomial order 3
    #Im_fit = savgol_filter(Im, 51, 5)
    return np.array(w), np.array(Im), np.array(Re)
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
        print("[broaden_data]: Ignoring imaginary modes!")
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
    #
    f = open("Intensity_"+str(w0)+"eV.dat",'w')
    f.write("# Raman intensity at "+str(w0)+"eV laser-wavelength\n")
    f.write("# freq/cm-1        xx         yy          zz        xy        yz        xz       avg\n")
    np.savetxt(f, tmp)
    f.close()
#

"""
def rotate_alpha(file, matrix):
    # rotate the Raman Intensity by matrix
    data = np.genfromtxt(file)
    w = [x[0] for x in data]
    x11 = [x[1] for x in data]
    x22 = [x[2] for x in data]
    x33 = [x[3] for x in data]
    x12 = [x[4] for x in data]
    x13 = [x[6] for x in data]
    x23 = [x[5] for x in data]

    orig = np.array([[x11, x12, x13],[x12, x22, x23], [x13, x23, x33]])
    new = np.empty_like(orig)
    for j in range(len(x11)):
        new[:,:,j] = np.dot(matrix.transpose(), np.dot(orig[:,:,j], matrix))
    #
    # write to file
    f = open(file+"_rotated",'w')
    f.write("# Raman intensity at ?eV laser-wavelength\n")
    f.write("# freq/cm-1        xx         yy          zz        xy        yz        xz       avg\n")
    tmp = np.empty_like(data)
    for j in range(len(x11)):
        tmp[j,0] = w[j]
        tmp[j,1] = new[0,0,j]
        tmp[j,2] = new[1,1,j]
        tmp[j,3] = new[2,2,j]
        tmp[j,4] = new[0,1,j]
        tmp[j,5] = new[1,2,j]
        tmp[j,6] = new[0,2,j]
    #
    np.savetxt(f, tmp)
    f.close()
"""
    
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

"""
#the stuff for the chi2 correction of LO modes, does not work!!
def get_born_from_vasprunxml(xml_fh, nat):
    xml_fh.seek(0)
    while True:
        line = xml_fh.readline()
        if not line:
            break
        #
        if "<array name=\"born_charges\" >" in line:
            born = np.zeros((nat,3,3))
            xml_fh.readline()# <dimension dim="1">ion</dimension>
            #
            for i in range(nat):
                xml_fh.readline() # <set>
                for j in range(3):
                    line = xml_fh.readline().split()
                    born[i,j] = [float(line[1]), float(line[2]), float(line[3])]
                #
                xml_fh.readline() # <\set>
            #
            print("[get_born_from_vasprunxml]: Read BORN from IR/vasprun.xml")
            #format: born[ION][LINE][COLUMN]
            return born
        #
    print("[get_born_from_vasprunxml]: WARNING Couldn't find 'born_charges' in vasprun.xml. Continuing...")
#

def get_dielectric_tensor_from_OUTCAR(outcar_fh):
# collect dielectric tensor from OUTCAR
    outcar_fh.seek(0)
    while True:
        line = outcar_fh.readline()
        if not line:
            break
        #
        if "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)" in line:
            dielectric = np.zeros((3,3))
            outcar_fh.readline() # ----------------------------------------------------
            #
            for j in range(3):
                line = outcar_fh.readline().split()
                dielectric[j] = [float(line[0]), float(line[1]), float(line[2])]
            #
            print("[get_dielectric_tensor_from_OUTCAR]: Read dielectric tensor from OUTCAR")
            #format: dielectric[LINE][COLUMN]
            return dielectric
        #
    print("[get_dielectric_tensor_from_OUTCAR]: WARNING Couldn't find 'MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)' in OUTCAR. Continuing...")
#

def get_chi2():
    xx = np.genfromtxt("oxx", dtype=float)
    xy = np.genfromtxt("oxy", dtype=float)
    xz = np.genfromtxt("oxz", dtype=float)
    yy = np.genfromtxt("oyy", dtype=float)
    yz = np.genfromtxt("oyz", dtype=float)
    zz = np.genfromtxt("ozz", dtype=float)
    return [xx, yy, zz, xy, yz, xz]

def get_LO_correction(chi2_tmp, born, eps_inf, qdir, vol, eigvec, w):
    lenw = len(w)
    chi2 = np.empty((3,3,3,lenw), dtype=complex)
    
    chi2[0,0,0] = np.interp(w, [x[0] for x in chi2_tmp[0]], [complex(x[2], x[1]) for x in chi2_tmp[0]])
    chi2[0,0,1] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[2], x[1]) for x in chi2_tmp[3]])
    chi2[0,0,2] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[2], x[1]) for x in chi2_tmp[5]])
    chi2[1,0,0] = np.interp(w, [x[0] for x in chi2_tmp[0]], [complex(x[4], x[3]) for x in chi2_tmp[0]])
    chi2[1,0,1] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[4], x[3]) for x in chi2_tmp[3]])
    chi2[1,0,2] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[4], x[3]) for x in chi2_tmp[5]])
    chi2[2,0,0] = np.interp(w, [x[0] for x in chi2_tmp[0]], [complex(x[6], x[5]) for x in chi2_tmp[0]])
    chi2[2,0,1] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[6], x[5]) for x in chi2_tmp[3]])
    chi2[2,0,2] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[6], x[5]) for x in chi2_tmp[5]])

    chi2[0,1,0] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[2], x[1]) for x in chi2_tmp[3]])
    chi2[0,1,1] = np.interp(w, [x[0] for x in chi2_tmp[1]], [complex(x[2], x[1]) for x in chi2_tmp[1]])
    chi2[0,1,2] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[2], x[1]) for x in chi2_tmp[4]])
    chi2[1,1,0] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[4], x[3]) for x in chi2_tmp[3]])
    chi2[1,1,1] = np.interp(w, [x[0] for x in chi2_tmp[1]], [complex(x[4], x[3]) for x in chi2_tmp[1]])
    chi2[1,1,2] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[4], x[3]) for x in chi2_tmp[4]])
    chi2[2,1,0] = np.interp(w, [x[0] for x in chi2_tmp[3]], [complex(x[6], x[5]) for x in chi2_tmp[3]])
    chi2[2,1,1] = np.interp(w, [x[0] for x in chi2_tmp[1]], [complex(x[6], x[5]) for x in chi2_tmp[1]])
    chi2[2,1,2] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[6], x[5]) for x in chi2_tmp[4]])

    chi2[0,2,0] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[2], x[1]) for x in chi2_tmp[5]])
    chi2[0,2,1] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[2], x[1]) for x in chi2_tmp[4]])
    chi2[0,2,2] = np.interp(w, [x[0] for x in chi2_tmp[2]], [complex(x[2], x[1]) for x in chi2_tmp[2]])
    chi2[1,2,0] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[4], x[3]) for x in chi2_tmp[5]])
    chi2[1,2,1] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[4], x[3]) for x in chi2_tmp[4]])
    chi2[1,2,2] = np.interp(w, [x[0] for x in chi2_tmp[2]], [complex(x[4], x[3]) for x in chi2_tmp[2]])
    chi2[2,2,0] = np.interp(w, [x[0] for x in chi2_tmp[5]], [complex(x[6], x[5]) for x in chi2_tmp[5]])
    chi2[2,2,1] = np.interp(w, [x[0] for x in chi2_tmp[4]], [complex(x[6], x[5]) for x in chi2_tmp[4]])
    chi2[2,2,2] = np.interp(w, [x[0] for x in chi2_tmp[2]], [complex(x[6], x[5]) for x in chi2_tmp[2]])

    #
    # formats
    # chi2[i][j][l][w]
    # eigvec[atom][k]
    # born[atom][l][k]
    # qdir[l]
    # eps_inf[l][k]
    corr = np.empty((3,3,lenw), dtype=complex)
    for w in range(lenw):
        for atom in range(nat):
            for k in range(3):
                tmp1 = 0
                tmp2 = 0
                tmp3 = 0
                for l in range(3):
                    tmp1 += qdir[l]*born[atom,l,k]
                    tmp3 += chi2[:,:,l,w]*qdir[l]
                    for ls in range(3):
                        tmp2 += qdir[l]*eps_inf[l,ls]*qdir[ls]
                    #
                #               angst^-3        e        yambo    angst?
                corr[:,:,w] += 8*np.pi/vol * tmp1/tmp2 * tmp3 * eigvec[atom][k]
            #
        #
    #                e->As               angst -> m   yambo -> pm/V     pm -> m       eps0
    corr = corr * 1.60217663+10**(-19) * 10**(-20) / 2.38721*10**(-9) * 10**(-12) / 8.8541878176*10**(-12)# unit: 1/m 
    return np.array([corr[0,0,:], corr[1,1,:], corr[2,2,:], corr[0,1,:], corr[1,2,:], corr[0,2,:]])
#
"""

