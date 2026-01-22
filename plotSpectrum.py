#!/usr/bin/env python

#
# plotting script
#

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from RamanLib import portoq

def plotSpectrum(w0, porto, qdir):
    print("[plotSpectrum]: Plotting Raman spectrum")

    if qdir == (1,0,0):
        ki = "x"
        ko = "-x"
    elif qdir == (0,1,0):
        ki = "y"
        ko = "-y"
    elif qdir == (0,0,1):
        ki = "z"
        ko = "-z"
    elif qdir == (1,1,0):
        ki = "x"
        ko = "y"
    elif qdir == (0,1,1):
        ki = "y"
        ko = "z"
    elif qdir == (1,0,1):
        ki = "x"
        ko = "z"
    else:
        print("[plotSpectrum]: ERROR: invalid propagation direction specified, exiting...")
        sys.exit(1)
    #
    
    if str(porto) == "yx":
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
        
    # Fonts
    plt.rcParams.update({
        "text.usetex": True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "lualatex",
    })
    mpl.use('pgf')
    
    size = 12
    mpl.rcParams['font.size'] = size
    mpl.rcParams['axes.titlesize'] = size
    mpl.rcParams['axes.labelsize'] = size
    mpl.rcParams['xtick.labelsize'] = size
    mpl.rcParams['ytick.labelsize'] = size
    mpl.rcParams['legend.fontsize'] = size
    mpl.rcParams['figure.titlesize'] = size

    dft_raw_data = np.loadtxt("Intensity_"+str(w0)+"eV.dat") # format: wavelength (cm-1) Intensity
    dict = {"xx": 1, "yy": 2, "zz": 3, "xy": 4, "yz": 5, "xz": 6, "avg": 7}
    x_data = [x[0] for x in dft_raw_data]
    y_data = [x[dict[porto]] for x in dft_raw_data]
    ymax = np.max(y_data)
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
    ax.plot(x_data, y_data, color="black", label="")
    #ax.legend(fontsize=fontsize)
    ax.set_title("Raman: "+kis+"("+str(porto)+")"+kos+" polarization")
    ax.set_ylim([0, 1.1*ymax])
    ax.set_xlim([x_data[0], x_data[-1]])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (m$^2$/sr)", fontsize=size)
    plt.savefig("Raman_"+str(ki)+str(porto)+str(ko)+"_"+str(w0)+"eV.pdf")
    print("[plotSpectrum]: Done.")
    sys.exit(1)    
#