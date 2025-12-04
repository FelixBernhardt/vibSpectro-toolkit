#!/usr/bin/env python

#
# plotting script
#

import sys
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.patches as mpatches
from RamanLib import *

def plotSpectrum(w0, porto):
    print("[plotSpectrum]: Plotting Raman spectrum")
    
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
    print("[plotSpectrum]: plotting .("+porto+"). configuration")
        
    fontsize=12
    dft_raw_data = np.loadtxt("Intensity_"+str(porto)+".dat") # format: wavelength (cm-1) Intensity
    x_data = [x[0] for x in dft_raw_data]
    y_data = [x[1] for x in dft_raw_data]
    y_max = np.max(y_data)

    # print the peak positions
    indices = find_peaks(y_data/y_max, height=0.0001, width=1)
    for i in indices[0]:
        print(int(np.rint(x_data[i])), y_data[i]/y_max)
    # 

    # plot
    ax = plt.subplot()
    ax.plot(x_data, y_data/y_max, color="black", label="DFT")
    ax.legend(fontsize=fontsize)
    ax.set_title("Raman: a("+str(porto)+")$\\overline{\\rm{a}}$ polarization")
    ax.set_xlim(0,1000)
    ax.set_ylim(0,1.1)
    ax.set_yticks([])
    plt.xticks([0, 200, 400, 600, 800, 1000], labels=None, fontsize=fontsize)
    ax.set_xticklabels([0, 200, 400, 600, 800, 1000])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (arb. units)", fontsize=fontsize)
    plt.savefig("Raman_a"+str(porto)+"a.pdf")
    print("[plotSpectrum]: Done.")
    sys.exit(1)    
#