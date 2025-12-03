#!/usr/bin/env python

#
# generate cells along eigenmodes
# (run calculations separately for all cells)
# read in data from different codes
# calculate resonant Raman intensity for various modes
# i.e. two dielectric functions per mode
#

import sys
import os.path
import datetime
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib.patches as mpatches
from RamanLib import *
from displace import displace

sys.dont_write_bytecode = True

if __name__ == '__main__':

    print("")
    print("    VASP_Raman.py")
    print("")
    print("    Contributors: Felix Bernhardt  (JLU)")
    print("    Started at: "+datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print("")
    print("    options (mandatory):\n")
    print("        -g : generate POSCARs")
    print("        -c : calculate Raman Tensor")
    print("        -s : calculate Raman spectrum")
    print("        -p : plot Raman spectrum")
    print("        -LO: calculate LO frequencies")
    print("        -h : help message\n")
    print("    variables:\n")
    print("        nth: <modes_list>\n")
    print("")

    if len(sys.argv) > 1:
        if sys.argv[1] == "-g":
            print("[__main__]: Generating POSCARs")
            opt = "g"
        elif sys.argv[1] == "-c":
            print("[__main__]: Calculating Raman tensors")
            opt = "c"
        elif sys.argv[1] == "-s":
            print("[__main__]: Calculating Raman spectrum")
            opt = "s"
        #elif sys.argv[1] == "-LO":
        #    print("[__main__]: Calculating Raman spectrum using LO frequencies")
        #    opt = "LO"
        elif sys.argv[1] == "-p":
            print("[__main__]: Plotting Raman spectrum")
            opt = "p"
        elif sys.argv[1] == "-h":
            print("[__main__]: Printing script usage information")
            opt = "h"
        else:
            print("[__main__]: no valid option specified, exiting...")
            sys.exit(1)
        #
        """
        # check for LO 
        LO_flag = False
        if len(sys.argv) > 2 and opt == "LO":
            if sys.argv[2] == "-g":
                print("[__main__]: Generating POSCARs")
                opt2 = "g"
            elif sys.argv[2] == "-c":
                print("[__main__]: Calculating Raman tensors")
                opt2 = "c"
            elif sys.argv[2] == "-s":
                print("[__main__]: Calculating Raman spectrum")
                opt2 = "s"
            elif sys.argv[2] == "-p":
                print("[__main__]: Plotting Raman spectrum")
                opt2 = "p"
            else:
                print("[__main__]: no valid option specified, exiting...")
                sys.exit(1)
            #
            qpoint = []
            count = 0
            for j in sys.argv[3:]:
                count += 1
                if count == 4:
                    break
                elif j.lstrip("-").isdigit() == True:
                    qpoint.append(int(j))
                else:
                    print("[__main__]: I don't understand the q-direction, exiting...")
                    sys.exit(1)
                #
            #
            qfile = "qpoints_"+str(qpoint[0])+"."+str(qpoint[1])+"."+str(qpoint[2])+".yaml"
        if ( len(sys.argv) > 2 and ( opt=="g" or opt=="c" or opt=="s" )) or len(sys.argv) > 6 and opt == "LO":
            if opt == "LO":
                index = 4
            else:
                index = 0
        """
        # check the modes
        modeList = []
        for j in sys.argv[2:]:
            if j.isdigit() == True:
                modeList.append(int(j))
            elif j.split("-")[0].isdigit() == True and j.split("-")[-1].isdigit() == True:
                if int(j.split("-")[0]) < int(j.split("-")[-1]):
                    for k in range(int(j.split("-")[0]),int(j.split("-")[-1])+1):
                        modeList.append(int(k))
                    #
                else:
                    print("[__main__]: First limit of range has to be SMALLER than second, exiting...")
                    sys.exit(1)
                    #
            else:
                print("[__main__]: I don't understand which modes you want to have put out, exiting...")
                sys.exit(1)
            #
        #
        modeList.sort()
        modeList = list(dict.fromkeys(modeList))
        print("[__main__]: Calculating modes " + str(modeList))
        #
    #
    else:
        print("[__main__]: no valid option specified, exiting...")
        sys.exit(1)  
    #

    if opt == "h":
        print("A single script to calculate Raman spectra with VASP")
        print("needs following files:")
        print("- POSCAR.phon     : the equilibrium structure, for which all modes are already calculated")
        print("- OUTCAR.phon     : VASP output containing the phonon eigenmodes (after division by sqrt(mass))")
        print("                    You have to have used NWRITE=3 in this phonon calculation!")
        print("- vasprun_phon.xml: VASP output containing the BORN charges (for LO-TO split only)\n")

        print("Format of <modes_list>:")
        print("- sinlge modes: integers separated by blank ' '")
        print("- range of modes: lower and upper limit separated by hyphen '-'")
        print("- example: '3-6 8 11-15' will calculate modes 3, 4, 5, 6, 8, 11, 12, 13, 14, 15")

        print("Calculation steps:")
        print("1  : run 'VASP_Raman.py -g <modes_list>")
        print("2  : run VASP for all the generated POSCARs")
        print("3  : rename the resulting vasprun.xml files to vasprun_<mode>_1 or vasprun_<mode>_-1")
        print("4  : run VASP_Raman.py -c <modes_list>")
        print("     this will calculate the LO mode frequencies and write them in the header of the Raman tensors")
        print("5.1: run VASP_Raman.py -s <modes_list>")
        print("5.2: set the parameters for the Raman-intensity calculation")
        print("     excitation frequency wavelength (eV), temperature (K), smearing width (cm-1)")
        print("optional steps:")
        print("6.1: run 'VASP_Raman.py -p")
        print("6.2: set the polarization direction (Porto notation, i.e. xx)") 
        print("     check the selection rules to correctly include LO-TO splitting if present")
        print("     https://www.cryst.ehu.es/cryst/polarizationselrules.html")
        #print("7  : LO-TO splitting requires phonopy v2.7.0!")
        #print("7.1: prepare 'vasprun_phon.xml' (same calculation as for OUTCAR.phon),")
        #print("     and 'BORN' (file structure as required by phonopy!) ")
        #print("7.2: run VASP_Raman.py -LO a b c <modes_list>")
        #print("     where a b c denote the reciprocal k-direction to consider")
        #print("7.3: calculate the Raman spectra again with these LO frequencies (steps 5-6)")
        sys.exit(1)
    #
    """
    if opt == "LO":
        # prepare FORCE_CONSTANTS with phonopy=2.2x.x, does not work with e.g. 2.7.0 ????
        # calculae the dynamical matrix with phonopy=2.7.0 !!
        #os.system("phonopy --fc vasprun_phon.xml -c POSCAR.phon -q")
        #with open("BORN", "w") as f:
        #    f.write(os.popen("phonopy-vasp-born vasprun_phon.xml").read())
        #
        # get lim q->0 for q-direction
        qdir = [q*0.0001 for q in qpoint]
        print("[__main__]: q-point =", qdir)
        os.system("phonopy --readfc -c POSCAR.phon --writedm --nac --q-direction=\""+str(qpoint[0])+" "+str(qpoint[1])+" "+str(qpoint[2])+"\" --qpoints=\"0 0 0\" --dim=\"1 1 1\" -q")
        # save the LO frequencies for later
        os.system("mv qpoints.yaml "+qfile)

        # get lim q->0 for q-direction, without BORN (i.e. without LO-TO splitting)
        os.system("phonopy --readfc -c POSCAR.phon --writedm --qpoints=\"0 0 0\" --dim=\"1 1 1\" -q")
        os.system("mv qpoints.yaml "+qfile+"0")

        poscar_fh = open("POSCAR.phon", 'r')
        nat, vol, b, pos, poscar_header, num_atoms, atom_types = parse_poscar(poscar_fh)
        poscar_fh.close()
        eigvals_ph, eigvecs_ph = parse_phonopy(qfile)
        eigvals, eigvecs_0 = parse_phonopy(qfile+"0")
        # get LO-TO splitting at q
        mode_dict = phonopy_assign(eigvecs_0, eigvals, eigvecs_ph, eigvals_ph, nat)
        # now set the LO-frequencies from phonopy onto the TO-eigenmodes (at Gamma) from VASP
        for j in range(3*nat):
            eigvals[j] = eigvals_ph[mode_dict[j]]
        #
        print(eigvals)
        eigvecs = phonopy2disp(eigvecs_ph, atom_types, num_atoms)
        norms = np.empty(3*nat)
        for i in range(3*nat):
            norms[i] = sqrt( sum( [abs(x)**2 for sublist in eigvecs[i] for x in sublist] ) )
        #
        LO_dir = str(qpoint[0])+str(qpoint[1])+str(qpoint[2])
        LO_dir = ""
        
        # get additional quantities needed for the modified raman tensor
        vasprun_fh = open("vasprun_phon.xml", "r")
        born = get_born_from_vasprunxml(vasprun_fh, nat)
        vasprun_fh.close()
        outcar_fh = open("OUTCAR.phon", "r")
        eps_inf = get_dielectric_tensor_from_OUTCAR(outcar_fh)
        outcar_fh.close()
        chi2 = get_chi2()

        LO_flag = True
        # continue with the LO modes and frequencies
        opt = opt2
    """
    #
    stepSize = 0.01 # Hardcoded, should be fine for almost all systems, also hardcoded in calc_raman
    disps = [-1, 1]
    programIN = "VASP"
    programOUT = "VASP"

    if opt == "g":
        displace(modeList, disps, stepSize, programIN, programOUT)
    
    """
    # TODO
    elif opt == "c":
        if LO_flag == False:
            LO_dir = ""
            poscar_fh = open("POSCAR.phon", 'r')
            nat, vol, b, pos, poscar_header, num_atoms, atom_types = parse_poscar(poscar_fh)
            poscar_fh.close()
            outcar_fh = open('OUTCAR.phon', 'r')
            eigvals, eigvecs, norms = get_modes_from_OUTCAR(outcar_fh, nat)
            outcar_fh.close()
        #
        print("[__main__]: Calculating Raman tensors...")
        #file_check(modeList, "vasprun", LO_dir)
        breakout = 0
        iteration = 0
        total = len(modeList)
        for mode in modeList:
            printProgressBar(iteration, total-1)
            eigval = eigvals[mode-1]
            eigvec = eigvecs[mode-1]
            norm = norms[mode-1]
            breakout = grep_optics("vasprun"+LO_dir+""+str(mode)+"_1", breakout)
            w1, Im1, Re1 = read_optics("optics.dat")
            breakout = grep_optics("vasprun"+LO_dir+""+str(mode)+"_-1", breakout)
            w2, Im2, Re2 = read_optics("optics.dat")
            if breakout == 0:
                #print("[__main__]: Calculating mode "+str(mode))
                w, Im1, Re1, Im2, Re2 = align_omega(w1, w2, Im1, Re1, Im2, Re2)
                #if LO_flag == True:
                #    LO_corr = get_LO_correction(chi2, born, eps_inf, qdir, vol, eigvec, w)
                LO_corr = None
                #
                calc_raman(mode, eigval, w, Im1, Re1, Im2, Re2, LO_corr, LO_dir)
            #
            iteration += 1
        #
        os.system("rm grep_optics.sh")
        os.system("rm optics.dat")
        if breakout == 1:
            sys.exit(1)
        #
        print("[__main__]: Done.")
        sys.exit(1)
    #
    elif opt == "s":
        if LO_flag == False:
            LO_dir = ""
            poscar_fh = open("POSCAR.phon", 'r')
            nat, vol, b, pos, poscar_header, num_atoms, atom_types = parse_poscar(poscar_fh)
            poscar_fh.close()
            outcar_fh = open('OUTCAR.phon', 'r')
            eigvals, eigvecs, norms = get_modes_from_OUTCAR(outcar_fh, nat)
            outcar_fh.close()
        #
        print("[__main__]: Calculating Raman spectrum")
        print("[__main__]: Note: check e.g. https://www.cryst.ehu.es/cryst/polarizationselrules.html for selection rules")
        w0 = 2.0
        temp = 300.0
        smear = 5.0
        #w0 = float(input("[__main__]: enter laser wavelength    (eV): "))
        #temp = float(input("[__main__]: enter temperature         (K) : "))
        #smear = float(input("[__main__]: enter smearing width    (cm-1): "))
        dict = {0: 'xx', 1: 'yy', 2: 'zz', 3: 'xy', 4: 'yz', 5: 'xz', 6: 'avg'}
        filelist = []

        file_check(modeList, "alpha", LO_dir)
        for mode in modeList:
            # ignore acoustic modes, rough!!
            if abs(eigvals[mode-1]) > 10:
                filelist.append("alpha"+LO_dir+"_"+str(mode)+".dat")
            #
        #
        # write Raman tensor for all modes at laser-wavelength w0
        print("[__main__]: Writing Raman"+LO_dir+"_"+str(w0)+"eV.dat")
        write_raman(filelist, w0, LO_dir)
        #
        print("[__main__]: Broadening spectrum")
        for col in range(7):
            broaden_data("Raman"+LO_dir+"_"+str(w0)+"eV.dat", w0, col, temp, smear)
        #
        cat_broaden(w0, LO_dir)
        print("[__main__]: Done.")
        sys.exit(1)
    #
    elif opt == "p":
        if LO_flag == False:
            LO_dir = ""
        print("[__main__]: Plotting Raman spectrum")
        porto = input("[__main__]: enter polarizations to plot (i.e. yy instead of x(yy)x), or avg:")

        if str(porto) == "yx":
            porto = "xy"
        elif str(porto) == "zy":
            porto = "yz"
        elif str(porto) == "zx":
            porto = "xz"
        elif str(porto) != "xx" and str(porto) != "yy" and str(porto) != "zz" and str(porto) != "xy" and str(porto) != "xz" and str(porto) != "yz" and str(porto) != "avg":
            print("[__main__]: ERROR: invalid polarization direction specified, exiting...")
            sys.exit(1)
        #
        print("[__main__]: Plotting Raman spectrum in "+str(porto)+" direction")
        
        fontsize=12
        dft_raw_data = np.loadtxt("Intensity"+LO_dir+"_"+str(porto)+".dat") # format: wavelength (cm-1) Intensity
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
        print("[__main__]: Done.")
        sys.exit(1)    
    #
    else:
        print("[__main__]: You should never have come here...")
        sys.exit(1)
    #
    """
#
