#!/usr/bin/env python

#
# plotting script
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plotSpectrum(path, w0, porto, qdir, lualatex=False):

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
        print("[plotSpectrum]: ERROR: invalid propagation direction specified")
        return 0
    #
    
    if str(porto) == "yx":
        porto = "xy"
    elif str(porto) == "zy":
        porto = "yz"
    elif str(porto) == "zx":
        porto = "xz"
    elif str(porto) != "xx" and str(porto) != "yy" and str(porto) != "zz" and str(porto) != "xy" and str(porto) != "xz" and str(porto) != "yz" and str(porto) != "perp" and str(porto) != "back":
        print("[plotSpectrum]: ERROR: invalid polarization direction specified, exiting")
        return 0
    #

    print("[plotSpectrum]: plotting "+ki+"("+porto+")"+ko+" configuration")
        
    # Fonts
    if lualatex == True:
        plt.rcParams.update({
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.texsystem": "lualatex",
        })
        mpl.use('pgf')
    #
    
    fig_width = 5.511 # inch
    mpl.rcParams['figure.figsize'] = [fig_width, fig_width/2]
    size = 12
    mpl.rcParams['font.size'] = size
    mpl.rcParams['axes.titlesize'] = size
    mpl.rcParams['axes.labelsize'] = size
    mpl.rcParams['xtick.labelsize'] = size
    mpl.rcParams['ytick.labelsize'] = size
    mpl.rcParams['legend.fontsize'] = size
    mpl.rcParams['figure.titlesize'] = size

    dft_raw_data = np.loadtxt(path+"Intensity_"+str(w0)+"eV.dat") # format: wavelength (cm-1) Intensity
    dict = {"xx": 1, "yy": 2, "zz": 3, "xy": 4, "yz": 5, "xz": 6, "perp": 7, "back": 8}
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
    ax.set_ylim(0, 1.1*ymax)
    ax.set_xlim(x_data[0], x_data[-1])
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Intensity (m$^2$/sr)", fontsize=size)
    plt.savefig(path+"Raman_"+str(ki)+str(porto)+str(ko)+"_"+str(w0)+"eV.pdf")
    plt.close()
#

def plotIRspectrum(path, file, lualatex=False):
    # data
    epsi_data = []
    epsr_data = []
    dft_raw_data = np.loadtxt(path+file) # format: wavelength (cm-1) Intensity (Imag, Real)
    dict = {0: "x", 1: "y", 2: "z", 3: "avg"}
    w_data = [x[0] for x in dft_raw_data]
    epsi_data.append( [x[1] for x in dft_raw_data] )
    epsr_data.append( [x[2] for x in dft_raw_data] )
    epsi_data.append( [x[3] for x in dft_raw_data] )
    epsr_data.append( [x[4] for x in dft_raw_data] )
    epsi_data.append( [x[5] for x in dft_raw_data] )
    epsr_data.append( [x[6] for x in dft_raw_data] )
    epsi_data.append( [x[7] for x in dft_raw_data] )
    epsr_data.append( [x[8] for x in dft_raw_data] )


    # Fonts
    if lualatex == True:
        plt.rcParams.update({
            "text.usetex": True,
            "pgf.rcfonts": False,
            "pgf.texsystem": "lualatex",
        })
        mpl.use('pgf')
    #
    fig_width = 5.511 # inch
    mpl.rcParams['figure.figsize'] = [fig_width, fig_width/2]
    size = 12
    mpl.rcParams['font.size'] = size
    mpl.rcParams['axes.titlesize'] = size
    mpl.rcParams['axes.labelsize'] = size
    mpl.rcParams['xtick.labelsize'] = size
    mpl.rcParams['ytick.labelsize'] = size
    mpl.rcParams['legend.fontsize'] = size
    mpl.rcParams['figure.titlesize'] = size

    for j in range(4):
        # plotting    
        fig, ((ax1, ax2)) = plt.subplots(1,2, layout="constrained")
        if j == 3:
            fig.suptitle("spatially averaged polarization")
        else:
            fig.suptitle("IR: E||"+dict[j]+" polarization")
        #
        ax1.set_xlim([w_data[0], w_data[-1]])
        ax1.plot(w_data, epsr_data[j], color="black", label="Real")
        ax1.axhline(ls="dashed")
        ax1.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax1.set_ylabel("Re($\\varepsilon$)")

        ax2.set_xlim([w_data[0], w_data[-1]])
        ax2.set_ylim([0, np.max(epsi_data[j])*1.1])
        ax2.plot(w_data, epsi_data[j], color="black", label="Imag")
        ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax2.set_ylabel("Im($\\varepsilon$)")
    
        plt.savefig(path+"IR_"+dict[j]+".pdf")
        plt.close()
    #
    print("[plotIR]: Done.") 
#