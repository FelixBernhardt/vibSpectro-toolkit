#!/usr/bin/env python

#
# library for VASP_Raman.py
#

import sys
import os.path

import numpy as np
from RamanLib import file_check, write_raman, cat_broaden, broaden_data

def calcSpectrum(modeList, w0, temp, smear):
    print("[calcSpectrum]: Calculating Raman spectrum")
    print("[calcSpectrum]: Note: check e.g. https://www.cryst.ehu.es/cryst/polarizationselrules.html for selection rules")
    print("[calcSpectrum]: Laser frequency set to "+str(w0)+"eV")
    print("[calcSpectrum]: Temperature set to "+str(temp)+"K")
    print("[calcSpectrum]: Smearing width set to "+str(smear)+"cm^-1")

    dict = {0: 'xx', 1: 'yy', 2: 'zz', 3: 'xy', 4: 'yz', 5: 'xz', 6: 'avg'}
    filelist = []

    file_check(modeList, "alpha")
    for mode in modeList:
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