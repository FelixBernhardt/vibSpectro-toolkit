import numpy as np

####################
# write data in yaml-style format
####################
def data(Phonon):
    yamlData = {"System": Phonon.name,
        "Pointgroup": Phonon.pointgroup,
        "Lattice": Phonon.basis.tolist(),
        "Points": [Phonon.elements, Phonon.cartesian],
        #"Born": [self.elements, self.born]
    }

    return yamlData
#

def writeData(Phonon):
    lines = []
    lines.append("System: " + Phonon.name)
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
        lines.append("- [ {: .6f},  {: .6f},  {: .6f}]".format(Phonon.basis[j][0], Phonon.basis[j][1], Phonon.basis[j][2]))
    lines.append("Points:")
    for atom in range(Phonon._nat):
        lines.append("- Symbol: "+Phonon.elements[atom]+" # "+str(atom+1))
        lines.append("  Coordinates: [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.direct[atom][0], Phonon.direct[atom][1], Phonon.direct[atom][2]))
        lines.append("  Mass: {: .6f}".format(Phonon.masses[atom]))
    
    lines.append("Modes:")
    for mode in Phonon.modelist:
        lines.append("- # "+str(mode)+" ("+Phonon.labels[mode-1] +"):")
        lines.append("- Frequency: {: .6f}".format(Phonon.eigenfreqs[mode-1]))
        for atom in range(Phonon._nat):
            lines.append("  - [{: .6f},  {: .6f},  {: .6f} ]".format(Phonon.eigenvecs[mode-1][atom][0], Phonon.eigenvecs[mode-1][atom][1], Phonon.eigenvecs[mode-1][atom][2]))
    
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

def writeRaman(Phonon):
    counter = 0
    if Phonon.code_out == "VASP":
        outfile = "vasprun.xml"
    elif Phonon.code_out == "QE":
        outfile = "epsilon.out"
    #

    for mode in Phonon.modelist:
        lines = []
        lines.append("System: " + Phonon.name)
        lines.append("Source: " + Phonon.path +"displacements/mode" + str(mode) + "_1/" + outfile)
        lines.append("        " + Phonon.path +"displacements/mode" + str(mode) + "_-1/" + outfile)
        lines.append("Mode: " + str(mode) + "(" + Phonon.labels[mode-1] + ")")
        lines.append("- Frequency: {: .6f}".format(Phonon.eigenfreqs[mode-1]))
        lines.append("- Raman_tensor:")
        
        for i in range(len(Phonon.ramantensors_data[counter])):
            lines.append("  - Laser_frequency: {: .6f}".format(Phonon.ramantensors_data[counter][i][0].real))
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[counter][i][1], Phonon.ramantensors_data[counter][i][4], Phonon.ramantensors_data[counter][i][6]))
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[counter][i][4], Phonon.ramantensors_data[counter][i][2], Phonon.ramantensors_data[counter][i][5]))
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[counter][i][6], Phonon.ramantensors_data[counter][i][5], Phonon.ramantensors_data[counter][i][3]))
            lines.append("    - perpendicular : {: .6f}".format(Phonon.ramantensors_data[counter][i][7].real))
            lines.append("    - backscattering: {: .6f}".format(Phonon.ramantensors_data[counter][i][8].real))

        counter += 1
        
        with open(Phonon.path+"Ramantensors/alpha"+str(mode)+".yaml", "w") as w:
            w.write("\n".join(lines))
        #
    #
#

def writeConstantRaman(Phonon):
    counter = 0
    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Source: " + Phonon.path+"Ramantensors/alpha*.yaml")
    lines.append("Laser_frequency: {: .6f}".format(Phonon.photon_freq))
    for mode in Phonon.modelist:
        lines.append("Mode: " + str(mode) + " (" + Phonon.labels[mode-1] + ")")
        lines.append("- Frequency: {: .6f}".format(Phonon.eigenfreqs[mode-1]))
        lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[counter][1], Phonon.constantraman_data[counter][4], Phonon.constantraman_data[counter][6]))
        lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[counter][4], Phonon.constantraman_data[counter][2], Phonon.constantraman_data[counter][5]))
        lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[counter][6], Phonon.constantraman_data[counter][5], Phonon.constantraman_data[counter][3]))
        lines.append("    - perpendicular : {: .6f}".format(Phonon.constantraman_data[counter][7].real))
        lines.append("    - backscattering: {: .6f}".format(Phonon.constantraman_data[counter][8].real))
        counter += 1
    
    with open(Phonon.path+"Raman.yaml", "w") as w:
        w.write("\n".join(lines))
    #
#

def writeSpectrum(Phonon):
    counter = 0
    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Source: " + Phonon.path+"Raman.yaml")
    lines.append("Laser_frequency: {: .6f}".format(Phonon.photon_freq))
    for polarization in ["xx", "yy", "zz", "xy", "yz", "xz", "perpendickular", "backscattering"]:
        lines.append("polarization: " + polarization)
        for j in range(len(Phonon.ramanspectrum_data[0][0])):
            lines.append("- [ {: .6f},  {: .6f} ]".format(Phonon.ramanspectrum_data[0][0][j], 1e30*Phonon.ramanspectrum_data[counter][1][j]))
        counter += 1
    
    with open(Phonon.path+"Intensity.yaml", "w") as w:
        w.write("\n".join(lines))
    #
#