import numpy as np
from Symmetries import portoq

####################
# write data in yaml-style format
####################
def writeData(Phonon):
    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Units:")
    lines.append("- Length: angstrom")
    lines.append("- Coordinates: direct")
    lines.append("- Mass: a.m.u")
    lines.append("- Frequency: cm⁻1")
    lines.append("- Phonon_Eigenvectors: cartesian")
    lines.append("- Charges: e")

    lines.append("Source: " +Phonon.path+Phonon.file)
    if Phonon.file == "phonopy.yaml":
        lines.append("        "+Phonon.path+"qpoints.yaml")
        if np.any(Phonon.born != 0):
            lines.append("        "+Phonon.path+"BORN")
    if Phonon.pointgroup == "":
        lines.append("Symmetry: Off")
    else:
        lines.append("        "+Phonon.path+"FORCE_CONSTANTS")
        lines.append("Pointgroup: "+Phonon.pointgroup) 
    
    lines.append("Lattice:") 
    for j in range(3):
        lines.append("- [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.basis[j][0], Phonon.basis[j][1], Phonon.basis[j][2]))
    lines.append("Points:")
    for atom in range(Phonon._nat):
        lines.append("- Symbol: "+Phonon.elements[atom]+" # "+str(atom+1))
        lines.append("  Coordinates: [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.direct[atom][0], Phonon.direct[atom][1], Phonon.direct[atom][2]))
        lines.append("  Mass: {: .6f}".format(Phonon.masses[atom]))
    
    lines.append("Modes:")
    for mode in Phonon.modelist:
        lines.append("- # "+str(mode)+" ("+Phonon.labels[mode] +"):")
        lines.append("- Frequency: {: .6f}".format(Phonon.eigenfreqs[mode]))
        for atom in range(Phonon._nat):
            lines.append("  - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.eigenvecs[mode][atom][0], Phonon.eigenvecs[mode][atom][1], Phonon.eigenvecs[mode][atom][2]))

    if np.any(Phonon.born != 0):
        lines.append("Born:")
        for atom in range(Phonon._nat):
            lines.append("- # "+str(atom+1)+" ("+Phonon.elements[atom]+")")
            for k in range(3):
                lines.append("  - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.born[atom][k][0], Phonon.born[atom][k][1], Phonon.born[atom][k][2]))
        lines.append("Dielectric_Constant:")
        for j in range(3):
            lines.append("- [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.eps_inf[j][0], Phonon.eps_inf[j][1], Phonon.eps_inf[j][2]))

    with open(Phonon.path+Phonon.name+".yaml", "w") as w:
        w.write("\n".join(lines))
    #
#

#######
# Raman
#######

def writeRaman(Phonon):
    if Phonon.code_out == "VASP":
        outfile = "vasprun.xml"
    elif Phonon.code_out == "QE":
        outfile = "epsilon.out"
    #

    for mode in Phonon.Ramanmodelist:
        lines = []
        lines.append("System: " + Phonon.name)
        lines.append("Source: " + Phonon.path +"displacements/mode" + str(mode) + "_1/" + outfile)
        lines.append("        " + Phonon.path +"displacements/mode" + str(mode) + "_-1/" + outfile)
        lines.append("Units:")
        lines.append("- Frequency: cm⁻1")
        lines.append("- Laser_Frequency: eV")
        lines.append("- Raman_Tensor: 10⁻30 Cm^2/V")
        lines.append("Mode: " + str(mode) + "(" + Phonon.labels[mode] + ")")
        lines.append("- Frequency: {: .6f}".format(Phonon.eigenfreqs[mode-1]))
        lines.append("- Raman_Tensor:")
        
        for i in range(len(Phonon.ramantensors_data[mode])):
            lines.append("  - Laser_Frequency: {: .6f}".format(Phonon.ramantensors_data[mode][i][0].real))
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[mode][i][1], Phonon.ramantensors_data[mode][i][4], Phonon.ramantensors_data[mode][i][6]))
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[mode][i][4], Phonon.ramantensors_data[mode][i][2], Phonon.ramantensors_data[mode][i][5]))
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[mode][i][6], Phonon.ramantensors_data[mode][i][5], Phonon.ramantensors_data[mode][i][3]))
            lines.append("    - perpendicular : {: .6f}".format(Phonon.ramantensors_data[mode][i][7].real))
            lines.append("    - backscattering: {: .6f}".format(Phonon.ramantensors_data[mode][i][8].real))
        
        with open(Phonon.path+"Ramantensors/alpha"+str(mode)+".yaml", "w") as w:
            w.write("\n".join(lines))
        #
    #
#

def writeConstantRaman(Phonon):
    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Source: " + Phonon.path+"Ramantensors/alpha*.yaml")
    lines.append("Units:")
    lines.append("- Frequency: cm⁻1")
    lines.append("- Laser_Frequency: eV")
    lines.append("- Raman_Tensor: 10⁻30 Cm^2/V")
    lines.append("Laser_frequency: {: .6f}".format(Phonon.photon_freq))
    for mode in Phonon.Ramanmodelist:
        lines.append("Mode: " + str(mode) + " (" + Phonon.labels[mode] + ")")
        lines.append("- Frequency: {: .6f}".format(Phonon.eigenfreqs[mode]))
        lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[mode][0], Phonon.constantraman_data[mode][3], Phonon.constantraman_data[mode][5]))
        lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[mode][3], Phonon.constantraman_data[mode][1], Phonon.constantraman_data[mode][4]))
        lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[mode][5], Phonon.constantraman_data[mode][4], Phonon.constantraman_data[mode][2]))
        lines.append("    - perpendicular : {: .6f}".format(Phonon.constantraman_data[mode][6].real))
        lines.append("    - backscattering: {: .6f}".format(Phonon.constantraman_data[mode][7].real))
    
    with open(Phonon.path+"Raman.yaml", "w") as w:
        w.write("\n".join(lines))
    #
#

def writeRamanSpectrum(Phonon):
    if Phonon.qdir == (1,0,0):
        ki = "x"
        ko = "-x"
    elif Phonon.qdir == (0,1,0):
        ki = "y"
        ko = "-y"
    elif Phonon.qdir == (0,0,1):
        ki = "z"
        ko = "-z"
    elif Phonon.qdir == (1,1,0):
        ki = "x"
        ko = "y"
    elif Phonon.qdir == (0,1,1):
        ki = "y"
        ko = "z"
    elif Phonon.qdir == (1,0,1):
        ki = "x"
        ko = "z"

    dict = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yx": 3, "yz": 4, "zy": 4, "xz": 5, "zx": 5, "perpendicular": 6, "backscattering": 7}

    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Source: " + Phonon.path + "Raman.yaml")
    if Phonon.LOcorr == True:
        lines.append("        " + Phonon.path + "qpoints_"+portoq[Phonon.qdir] + ".yaml")
        lines.append("        " + Phonon.path + "whateverFile")
    lines.append("Units:")
    lines.append("- Frequency: cm⁻1")
    lines.append("- Laser_Frequency: eV")
    lines.append("- Raman_Intensity: m²/sr")
    lines.append("Laser_frequency: {: .6f}".format(Phonon.photon_freq))
    for polarization in ["xx", "yy", "zz", "xy", "yx", "yz", "zy", "xz", "zx", "perpendicular", "backscattering"]:
        lines.append("Polarization: " + ki + "(" + polarization + ")" + ko )
        for j in range(len(Phonon.ramanspectrum_data[0][0])):
            lines.append("- [ {: .6f},  {: .6f} ]".format(Phonon.ramanspectrum_data[0][0][j], 1e30*Phonon.ramanspectrum_data[dict[polarization]][1][j]))
    
    with open(Phonon.path+"Intensity.yaml", "w") as w:
        w.write("\n".join(lines))
    #
#

####
# IR
####

def writeIRSpectrum(Phonon):
    dict = {"x": 1, "y": 2, "z": 3}
    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Units:")
    lines.append("- Frequency: cm⁻1")

    lines.append("Source: " +Phonon.path+Phonon.file)
    if Phonon.file == "phonopy.yaml":
        lines.append("        "+Phonon.path+"qpoints.yaml")
        lines.append("        "+Phonon.path+"BORN")

    for polarization in ["x", "y", "z"]:
        lines.append("Polarization: E || "  + polarization )
        for j in range(len(Phonon.IR_data[0])):
            lines.append("- [ {: .6f},  {: .6f} ]".format(Phonon.IR_data[0][j].real, Phonon.IR_data[dict[polarization]][j]))
    
    with open(Phonon.path+"IR.yaml", "w") as w:
        w.write("\n".join(lines))
    #
#

def writeReflectanceSpectrum(Phonon):
    dict = {"x": 1, "y": 2, "z": 3}
    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Units:")
    lines.append("- Frequency: cm⁻1")

    lines.append("Source: " +Phonon.path+Phonon.file)
    if Phonon.file == "phonopy.yaml":
        lines.append("        "+Phonon.path+"qpoints.yaml")
        lines.append("        "+Phonon.path+"BORN")

    for polarization in ["x", "y", "z"]:
        lines.append("Polarization: E || "  + polarization )
        for j in range(len(Phonon.reflectance_data[0])):
            lines.append("- [ {: .6f},  {: .6f} ]".format(Phonon.reflectance_data[0][j], Phonon.reflectance_data[dict[polarization]][j]))
    
    with open(Phonon.path+"Reflectance.yaml", "w") as w:
        w.write("\n".join(lines))
    #
#