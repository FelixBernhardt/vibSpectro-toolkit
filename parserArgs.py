#!/usr/bin/env python

#
# argument parser for Raman.py
#

import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--modeList", type=str, default="",\
                    help="The phonon modes to be considered for the calculations\n\
                    The labelling is in ascending order according to the mode's frequencies\
                    Format of modeList:\
                    . single modes: integers separated by blank \" \"\
                    - range of modes: lower and upper limit separated by hyphen \"-\"\
                     example: '3-6 8 11-15' will consider modes 3, 4, 5, 6, 8, 11, 12, 13, 14, 15")

parser.add_argument("-d", "--displace", action="store_true",\
                    help="add the ionic displacements according to the phonon mode..")
parser.add_argument("-t", "--tensors", action="store_true",\
                    help="Calculate the Raman tensors of the phonon modes provided by the --modeList option")
parser.add_argument("-s", "--spectrum", action="store_true",\
                    help="Calculate the Raman spectrum for back- and right angle scattering of all phonon modes provided by the --modeList option.")
parser.add_argument("-p", "--plot", action="store_true",\
                    help="plot the Raman spectrum of the configuration specified using the --porto option. Only the phonon modes considered for the calculation of the spectrum are considered.")

parser.add_argument("-w", "--laser", type=float, default=2.0,\
                    help="The excitation energy used for calculating the spectrum (eV)")
parser.add_argument("-tp", "--temperature", type=float, default=300.0,\
                    help="The temperature used for calculating the spectrum (K)")
parser.add_argument("-sm", "--smearing", type=float, default=5.0,\
                    help="The smearing used for calculating the spectrum")
parser.add_argument("-pt", "--porto", type=str, default="xx",\
                    help="The configuration used when plotting the spectrum using porto's notation\
                         .(xx). , where \"xx\" can be set to backscattering or right angle scattering.\
                         A spatially averaged spectrum can be plotted by setting to \"avg\".")

parser.add_argument("-PI", "--programIN", type=str, default="VASP",\
                    help="The software package to read phonon modes and structural information from")
parser.add_argument("-PO", "--programOUT", type=str, default="VASP",\
                    help="The software package to write the created cells into and read the dielectric function from")


args = parser.parse_args()

# check the modes
modeList = []
for j in args.modeList.split():
    if j.isdigit() == True:
        modeList.append(int(j))
    elif j.split("-")[0].isdigit() == True and j.split("-")[-1].isdigit() == True:
        if int(j.split("-")[0]) < int(j.split("-")[-1]):
            for k in range(int(j.split("-")[0]),int(j.split("-")[-1])+1):
                modeList.append(int(k))
            #
        else:
            print("[parserArgs]: First limit of modeList range has to be SMALLER than second, exiting...")
            sys.exit(1)
        #
    else:
        print("[parserArgs]: I don't understand which phonon modes you want to have put out, exiting...")
        sys.exit(1)
    #
#
modeList.sort()
args.modeList = list(dict.fromkeys(modeList))

# check the options
if args.modeList == [] and args.displace==True:
    print("[parserArgs]: Please provide modes for which the ions can be displaced, exiting...")
    sys.exit(1)
if args.modeList == [] and args.tensors==True:
    print("[parserArgs]: Please provide modes for which to calculate the Raman tensors, exiting...")
    sys.exit(1)
if args.modeList == [] and args.spectrum==True:
    print("[parserArgs]: Please provide modes for which to calculate the spectrum, exiting...")
    sys.exit(1)
#

"""
    if opt == "h":
        print("A single script to calculate Raman spectra with external DFT codes")
        print("needs following files (VASP used as an example):")
        print("- POSCAR.phon     : the equilibrium structure, for which all phonon modes and frequencies are already calculated")
        print("- OUTCAR.phon     : output containing the phonon eigenmodes and frequencies")

        print("Calculation steps:")
        print("1  : run 'Raman.py -d -m=<modeList>")
        print("2  : run VASP for all the generated POSCARs in their respective folders")
        print("4  : run Raman.py -t -m=<modeList>")
        print("     this will calculate the frequency dependent Raman tensors")
        print("5  : run Raman.py -s -m=<modeList> -w=<float> -tp=<float> -sm=<float>")
        print("     set the parameters for the Raman spectrum calculation")
        print("     excitation frequency wavelength (eV), temperature (K), smearing width (cm-1)")
        print("optional steps:")
        print("6    run 'Raman.py -p -pt=<str>")
        print("     set the polarization direction (Porto notation, i.e. xx)") 
        print("     check the selection rules to only include TO modes, e.g. here")
        print("     https://www.cryst.ehu.es/cryst/polarizationselrules.html")
        print("     the Raman intensities for LO-modes are not implemented yet!)
       
    #
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