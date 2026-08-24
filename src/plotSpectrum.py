#!/usr/bin/env python

#
# plotting script
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def qdir2ks(qdir):
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
        print("[qdir2ks]: ERROR: invalid propagation direction specified")
        return "a", "a"
    #
    return ki, ko
#

def _plotRamanSpectrum(ramandata, path, w0, porto, qdir, lualatex=False):

    ki, ko = qdir2ks(qdir)
    
    if str(porto) == "yx":
        porto = "xy"
    elif str(porto) == "zy":
        porto = "yz"
    elif str(porto) == "zx":
        porto = "xz"
    elif str(porto) != "xx" and str(porto) != "yy" and str(porto) != "zz" and str(porto) != "xy" and str(porto) != "xz" and str(porto) != "yz" and str(porto) != "perp" and str(porto) != "back":
        print("[_plotRamanSpectrum]: ERROR: invalid polarization direction specified, exiting")
        return 0
    #

    #print("[plotSpectrum]: plotting "+ki+"("+porto+")"+ko+" configuration")
        
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
    
    dict = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yz": 4, "xz": 5, "perp": 6, "back": 7}

    x_data = ramandata[dict[porto]][0]
    y_data = ramandata[dict[porto]][1]

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

def plotRamanSpectrum(ramanspectrum_data, path, photon_freq, qdir_cartesian, porto, lualatex):
    for pt in porto:
        _plotRamanSpectrum(ramanspectrum_data, path, photon_freq, pt, qdir_cartesian, lualatex)
    #
    print("[plotRamanSpectrum]: Done.") 
#

def rotation_matrix(rot_axis, theta):
    axis = rot_axis/np.linalg.norm(rot_axis)
    rot_matrix = np.zeros((3,3))
    rot_matrix[0,0] = axis[0]**2*(1-np.cos(theta)) + np.cos(theta)
    rot_matrix[1,1] = axis[1]**2*(1-np.cos(theta)) + np.cos(theta)
    rot_matrix[2,2] = axis[2]**2*(1-np.cos(theta)) + np.cos(theta)
    rot_matrix[0,1] = axis[0]*axis[1]*(1-np.cos(theta)) - axis[2]*np.sin(theta)
    rot_matrix[1,0] = axis[0]*axis[1]*(1-np.cos(theta)) + axis[2]*np.sin(theta)
    rot_matrix[0,2] = axis[0]*axis[2]*(1-np.cos(theta)) + axis[1]*np.sin(theta)
    rot_matrix[2,0] = axis[0]*axis[2]*(1-np.cos(theta)) - axis[1]*np.sin(theta)
    rot_matrix[1,2] = axis[1]*axis[2]*(1-np.cos(theta)) - axis[0]*np.sin(theta)
    rot_matrix[2,1] = axis[1]*axis[2]*(1-np.cos(theta)) + axis[0]*np.sin(theta)
    return rot_matrix
#

def plotPolarRaman(ramantensors_data, path, zero_axis, rotation_axis, modes, w0, qdir):

    print("[plotPolarRaman]: Mode: "+str(modes))
    print("[plotPolarRaman]: zero axis    : "+str(zero_axis))
    print("[plotPolarRaman]: rotation axis: "+str(rotation_axis))

    w_list = np.real(ramantensors_data[modes[0]][:,0])
    ki, ko = qdir2ks(qdir)

    if np.array_equal(zero_axis, np.array([1,0,0])):
        if np.array_equal(rotation_axis,np.array([0,1,0])):
            porto = "x-z"
        elif np.array_equal(rotation_axis, np.array([0,0,1])):
            porto = "x-y"
        else:
            porto ="x-"+str(rotation_axis)
        #
    elif np.array_equal(zero_axis, np.array([0,1,0])):
        if np.array_equal(rotation_axis, np.array([1,0,0])):
            porto = "y-z"
        elif np.array_equal(rotation_axis, np.array([0,0,1])):
            porto = "y-x"
        else:
            porto ="y-"+str(rotation_axis)
        #
    elif np.array_equal(zero_axis, np.array([0,0,1])):
        if np.array_equal(rotation_axis, np.array([1,0,0])):
            porto = "z-y"
        elif np.array_equal(rotation_axis, np.array([0,1,0])):
            porto = "z-x"
        else:
            porto ="z-"+str(rotation_axis)
        #
    else:
        porto = str(zero_axis)+"-"+str(rotation_axis)
    #

    theta = np.arange(0, 2*np.pi, 0.01*np.pi)
    rplot = []
    for mode in modes:

        ramantensor = np.zeros((3,3))
        ramantensor[0,0] = np.abs(np.interp([w0], w_list, ramantensors_data[mode][:,1]))[0]
        ramantensor[1,1] = np.abs(np.interp([w0], w_list, ramantensors_data[mode][:,2]))[0]
        ramantensor[2,2] = np.abs(np.interp([w0], w_list, ramantensors_data[mode][:,3]))[0]
        ramantensor[0,1] = np.abs(np.interp([w0], w_list, ramantensors_data[mode][:,4]))[0]
        ramantensor[1,2] = np.abs(np.interp([w0], w_list, ramantensors_data[mode][:,5]))[0]
        ramantensor[0,2] = np.abs(np.interp([w0], w_list, ramantensors_data[mode][:,6]))[0]
        ramantensor[1,0] = ramantensor[0,1]
        ramantensor[2,1] = ramantensor[1,2]
        ramantensor[2,0] = ramantensor[0,2]

        r = np.empty_like(theta)
        for i in range(len(theta)):
            r[i] = np.abs(np.dot(zero_axis, np.dot(ramantensor, np.dot(rotation_matrix(rotation_axis, theta[i]), zero_axis))))
        #
        rplot.append(r)
    #

    fig, ax = plt.subplots(1,1, subplot_kw={"projection": "polar"}, layout="constrained")

    for i in range(len(modes)):
        ax.plot(theta, rplot[i], label=modes[i])
    if np.max(rplot) < 1.0:
        ax.set_rmax(10)
    else:
        ax.set_rmax(np.max(rplot))
    #
    ax.set_rticks([])
    ax.grid(True)
    if len(modes) == 1:
        fig.suptitle("Mode "+str(modes)+", "+porto)
    else:
        ax.legend(loc="upper right", bbox_to_anchor=(1.6, 1))
        fig.suptitle("Modes "+str(modes)+", "+porto)
    plt.savefig(path+"Raman_"+str(ki)+str(porto)+str(ko)+"_"+str(w0)+"eV_mode"+str(modes)+".pdf")
    plt.close()
    print("[plotPolarRaman]: Done.")
#

def plotIRSpectrum(IRdata, path, file, lualatex=False):
    dict = {1: "x", 2: "y", 3: "z", 4: "avg"}
    w_data = IRdata[0].real

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

    for j in range(1,4):
        # plotting    
        fig, ((ax1, ax2)) = plt.subplots(1,2, layout="constrained")
        if j == 4:
            fig.suptitle("spatially averaged polarization")
        else:
            fig.suptitle("IR: E||"+dict[j]+" polarization")
        #
        ax1.set_xlim([w_data[0], w_data[-1]])
        ax1.plot(w_data, IRdata[j].real, color="black", label="Real")
        ax1.axhline(ls="dashed")
        ax1.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax1.set_ylabel("Re($\\varepsilon$)")

        ax2.set_xlim([w_data[0], w_data[-1]])
        ax2.set_ylim([0, np.max(IRdata[j].imag)*1.1])
        ax2.plot(w_data, IRdata[j].imag, color="black", label="Imag")
        ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax2.set_ylabel("Im($\\varepsilon$)")
    
        plt.savefig(path+"IR_"+dict[j]+".pdf")
        plt.close()
    #
    print("[plotIRSpectrum]: Done.") 
#

def plotReflectanceSpectrum(R_data, path, file, lualatex=False):
    dict = {1: "x", 2: "y", 3: "z", 4: "avg"}
    w_data = R_data[0]


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
    mpl.rcParams['figure.figsize'] = [fig_width, fig_width/1.2]
    size = 12
    mpl.rcParams['font.size'] = size
    mpl.rcParams['axes.titlesize'] = size
    mpl.rcParams['axes.labelsize'] = size
    mpl.rcParams['xtick.labelsize'] = size
    mpl.rcParams['ytick.labelsize'] = size
    mpl.rcParams['legend.fontsize'] = size
    mpl.rcParams['figure.titlesize'] = size

    for j in range(1,4):
        # plotting    
        fig, ax = plt.subplots(1,1, layout="constrained")
        fig.suptitle("Reflectance: E||"+dict[j]+" polarization")
        #
        ax.set_xlim([w_data[0], w_data[-1]])
        ax.set_ylim([0, 1.05])
        ax.plot(w_data, R_data[j], color="black", label="T")
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Reflectance")
    
        plt.savefig(path+"R_"+dict[j]+".pdf")
        plt.close()
    #
    print("[plotReflectanceSpectrum]: Done.") 
#