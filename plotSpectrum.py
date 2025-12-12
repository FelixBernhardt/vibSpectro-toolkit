#!/usr/bin/env python

#
# plotting script
#

import sys
import numpy as np
import matplotlib.pyplot as plt
from RamanLib import *
from parserPhonopy import parsePhonopy

def plotSpectrum(w0, porto):
    print("[plotSpectrum]: Plotting Raman spectrum")
    # use a dummy for modes to be considered
    freqs, eigvecs_new, norms, qpoint, basis, nat, elements, cPos, masses = parsePhonopy([1])

    if (qpoint[0] != 0.0 and qpoint[1] != 0.0 and qpoint[2] != 0.0) or (qpoint[0] == 0.0 and qpoint[1] == 0.0 and qpoint[2] == 0.0):
        ki = "K"
        ko = "K"
    
    if qpoint[0] > 0.0:
        ki = "x"
    elif qpoint[0] < 0.0:
        ki = "-x"
    if qpoint[1] > 0.0 and qpoint[0] != 0.0:
        ko = "y"
    elif qpoint[1] < 0.0 and qpoint[0] != 0.0:
        ko = "-y"
    if qpoint[1] > 0.0 and qpoint[2] != 0.0:
        ki = "y"
    elif qpoint[1] < 0.0 and qpoint[2] != 0.0:
        ki = "-y"
    if qpoint[2] > 0.0 and qpoint[0] != 0.0:
        ko = "z"
    elif qpoint[2] < 0.0 and qpoint[0] != 0.0:
        ko = "-z"
    if qpoint[2] > 0.0 and qpoint[1] != 0.0:
        ko = "z"
    elif qpoint[2] < 0.0 and qpoint[1] != 0.0:
        ko = "-z"
   
    

    if porto == None:
        porto = "xx"
    elif str(porto) == "yx":
        porto = "xy"
    elif str(porto) == "zy":
        porto = "yz"
    elif str(porto) == "zx":
        porto = "xz"
    elif str(porto) != "xx" and str(porto) != "yy" and str(porto) != "zz" and str(porto) != "xy" and str(porto) != "xz" and str(porto) != "yz" and str(porto) != "avg":
        print("[plotSpectrum]: ERROR: invalid polarization direction specified, exiting...")
        sys.exit(1)
        #
    print("[plotSpectrum]: plotting "+ki+"("+porto+")"+ko+" configuration")
        
    fontsize=12
    dft_raw_data = np.loadtxt("Intensity_"+str(porto)+".dat") # format: wavelength (cm-1) Intensity
    dict = {"xx": 1, "yy": 2, "zz": 3, "xy": 4, "yz": 5, "xz": 6, "avg": 7}
    x_data = [x[0] for x in dft_raw_data]
    y_data = [x[dict[porto]] for x in dft_raw_data]
    y_max = np.max(y_data)

    """
    # print the peak positions
    print("[plotSpectrum]: Found peaks at:")
    print("cm^-1    Intensity:")
    indices = find_peaks(y_data/y_max, height=0.0001, width=1)
    for i in indices[0]:
        print(int(np.rint(x_data[i])), y_data[i]/y_max)
    # 
    """

    # plot
    kis = ki
    kos = ko
    if ki[0] == "-":
        kis = "$\\overline{\\rm{"+ki[1]+"}}$"
    if ko[0] == "-":
        kos = "$\\overline{\\rm{"+ko[1]+"}}$"
    #
    ax = plt.subplot()
    ax.plot(x_data, y_data/y_max, color="black", label="")
    #ax.legend(fontsize=fontsize)
    ax.set_title("Raman: "+kis+"("+str(porto)+")"+kos+" polarization")
    #ax.set_xlim(0,1000)
    #ax.set_ylim(0,1.1)
    ax.set_yticks([])
    plt.xticks([0, 200, 400, 600, 800, 1000], labels=None, fontsize=fontsize)
    ax.set_xticklabels([0, 200, 400, 600, 800, 1000])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (arb. units)", fontsize=fontsize)
    plt.savefig("Raman_"+str(ki)+str(porto)+str(ko)+"_"+str(w0)+"eV.pdf")
    print("[plotSpectrum]: Done.")
    sys.exit(1)    
#