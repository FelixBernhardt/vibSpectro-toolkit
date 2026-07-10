"""Function for calculating the Raman spectrum for given Raman tensors."""
# Copyright (C) 2026 Felix Bernhardt
# All rights reserved.
#
# This file is part of <>.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the <> project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from src.Symmetries import eps0, c_cm, h, kb, ev2rcm

def Lorentz(hw, ab, gam=0.001):
    fmax = max(hw)
    erange = np.arange(0, 1.1*fmax, gam/10)
    spectrum = 0.0 * erange
    for i in range(len(hw)):
        spectrum +=  ab[i] * gam  / ( (hw[i]-erange)**2 + gam**2 )
    #
    return erange, spectrum
#

def broadenData(modelist, raman, eigvals, w0, col, temp, smear, stokes):
    # apply smearing to Raman tensors from "writeConstantRaman"
    
    # calculate the Raman intensity for each mode and component
    intensity = []
    for mode in modelist:
        cm1 = eigvals[mode]
        n  = (-np.exp(-h * cm1 * c_cm/(kb * temp))+1)**(-1)
        prefactor = h / (32 * np.pi**3 * (c_cm/100)**4 * eps0**2) * ( 2 * np.pi * c_cm )**3 * 10**(-30)
        # anti-stokes
        if stokes == False:
            intensity.append( prefactor*np.abs(raman[mode][col])**2 * (ev2rcm*w0 + cm1)**4 * (n-1)/cm1 )
        # Stokes
        else:
            intensity.append( prefactor*np.abs(raman[mode][col])**2 * (ev2rcm*w0 - cm1)**4 * n/cm1 )
        #
    #

    w, Spectrum = Lorentz([eigvals[mode] for mode in modelist], intensity, smear)

    return np.array([w, Spectrum])
#

def getConstantRaman(ramantensors, modelist, w0):
    Raman = {}
    for mode in modelist:
        w_list = np.real(ramantensors[mode][:,0])
        alpha = []

        for i in range(1, 9):
            alpha.append(np.abs(np.interp([w0], w_list, ramantensors[mode][:,i]))[0])
        #
        Raman[mode] = alpha
    #

    return Raman
#

def calcSpectrum(ramantensors, modelist, Ramanmodelist, eigvals, w0, temp, smear, stokes):
    print("[calcSpectrum]: Calculating Raman spectrum of modes "+str(modelist))
    #print("[calcSpectrum]: Note: check e.g. https://www.cryst.ehu.es/cryst/polarizationselrules.html for selection rules")
    print("[calcSpectrum]: Laser frequency set to "+str(w0)+"eV")
    print("[calcSpectrum]: Temperature set to "+str(temp)+"K")
    print("[calcSpectrum]: Smearing width set to "+str(smear)+"cm^-1")
    raman = getConstantRaman(ramantensors, Ramanmodelist, w0)
    spectrum = []
    for col in range(8):
        spectrum.append(broadenData(Ramanmodelist, raman, eigvals, w0, col, temp, smear, stokes))
    #
    print("[calcSpectrum]: Done.")
    return raman, np.array(spectrum)
#