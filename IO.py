import numpy as np
from Symmetries import portoq
import yaml

####################
# write system data in yaml-style format
####################
def writeData(Phonon):
    lines = []
    lines.append("System: \"" + Phonon.name+"\"")
    lines.append("Units:")
    lines.append(" Length: \"angstrom\"")
    lines.append(" Coordinates: \"direct\"")
    lines.append(" Mass: \"a.m.u\"")
    lines.append(" Frequency: \"cm⁻1\"")
    lines.append(" Phonon_Eigenvectors: \"cartesian\"")
    lines.append(" Charges: \"e\"")

    
    lines.append("Source:")
    lines.append("- \""+Phonon.path+Phonon.file+"\"")
    if Phonon.file == "phonopy.yaml":
        lines.append("- \""+Phonon.path+"qpoints.yaml\"")
        if np.any(Phonon.born != 0):
            lines.append(" \""+Phonon.path+"BORN\"")
    if Phonon.pointgroup == "":
        lines.append("Symmetry: \"Off\"")
    else:
        lines.append("- \""+Phonon.path+"FORCE_CONSTANTS\"")
        lines.append("Pointgroup: \""+Phonon.pointgroup+"\"") 
    
    lines.append("Lattice:") 
    for j in range(3):
        lines.append("- [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.basis[j][0], Phonon.basis[j][1], Phonon.basis[j][2]))
    
    lines.append("Points:")
    for atom in range(Phonon._nat):
        lines.append("- Symbol: "+Phonon.elements[atom]+" # "+str(atom+1))
        lines.append("  Coordinates: [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.direct[atom][0], Phonon.direct[atom][1], Phonon.direct[atom][2]))
        lines.append("  Mass: {: .6f}".format(Phonon.masses[atom]))
    
    lines.append("Modes:")
    for mode in Phonon._modelist:
        lines.append("- Mode: "+str(mode))
        lines.append("  Label: \"" + Phonon.labels[mode] +"\"")
        lines.append("  Frequency: {: .6f}".format(Phonon.eigenfreqs[mode]))
        lines.append("  Eigenvector:")
        for atom in range(Phonon._nat):
            lines.append("  - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.eigenvecs[mode][atom][0], Phonon.eigenvecs[mode][atom][1], Phonon.eigenvecs[mode][atom][2]))

    if np.any(Phonon.born != 0):
        lines.append("Born:")
        for atom in range(Phonon._nat):
            lines.append("- # "+str(atom+1)+" ("+Phonon.elements[atom]+")")
            for k in range(3):
                lines.append("  - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.born[atom][k][0], Phonon.born[atom][k][1], Phonon.born[atom][k][2]))
        lines.append("Dielectric_Tensor:")
        for j in range(3):
            lines.append("- [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.eps_inf[j][0], Phonon.eps_inf[j][1], Phonon.eps_inf[j][2]))
    
    with open(Phonon.path+Phonon.name+".yaml", "w") as w:
        w.write("\n".join(lines))
    #
#

#####################
# write Raman tensors
#####################

def writeRaman(Phonon):
    if Phonon.code_out == "VASP":
        outfile = "vasprun.xml"
    elif Phonon.code_out == "QE":
        outfile = "epsilon.out"
    else:
        outfile = Phonon.code_out
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

# Raman tensors at constant excitation energy
def writeConstantRaman(Phonon):
    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Source: " + Phonon.path+"Ramantensors/alpha*.yaml")
    lines.append("Units:")
    lines.append(" Frequency: cm⁻1")
    lines.append(" Laser_Frequency: eV")
    lines.append(" Raman_Tensor: 10⁻30 Cm^2/V")
    lines.append("Laser_frequency: {: .6f}".format(Phonon.photon_freq))
    lines.append("Modes:")
    for mode in Phonon.Ramanmodelist:
        lines.append("- Mode: " + str(mode))
        lines.append("  Label: \""+ Phonon.labels[mode] +"\"")
        lines.append("  Frequency: {: .6f}".format(Phonon.eigenfreqs[mode]))
        lines.append("  Tensor:")
        lines.append("  - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[mode][0], Phonon.constantraman_data[mode][3], Phonon.constantraman_data[mode][5]))
        lines.append("  - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[mode][3], Phonon.constantraman_data[mode][1], Phonon.constantraman_data[mode][4]))
        lines.append("  - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.constantraman_data[mode][5], Phonon.constantraman_data[mode][4], Phonon.constantraman_data[mode][2]))
        lines.append("  - perpendicular : {: .6f}".format(Phonon.constantraman_data[mode][6].real))
        lines.append("  - backscattering: {: .6f}".format(Phonon.constantraman_data[mode][7].real))
    
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
    else:
        ki = "x"
        ko = "x"
    #

    dict = {"xx": 0, "yy": 1, "zz": 2, "xy": 3, "yx": 3, "yz": 4, "zy": 4, "xz": 5, "zx": 5, "perpendicular": 6, "backscattering": 7}

    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Source: " + Phonon.path + "Raman.yaml")
    if Phonon.LOcorr == True:
        lines.append("        " + Phonon.path + "qpoints_"+portoq[Phonon.qdir] + ".yaml")
        lines.append("        " + Phonon.path + "whateverFile")
    lines.append("Units:")
    lines.append(" Frequency: cm⁻1")
    lines.append(" Laser_Frequency: eV")
    lines.append(" Raman_Intensity: m²/sr")
    lines.append("Laser_frequency: {: .6f}".format(Phonon.photon_freq))
    lines.append("Geometry:")
    for polarization in ["xx", "yy", "zz", "xy", "yx", "yz", "zy", "xz", "zx", "perpendicular", "backscattering"]:
        lines.append("- Polarization: " + ki + "(" + polarization + ")" + ko )
        lines.append("  Spectrum:")
        for j in range(len(Phonon.ramanspectrum_data[0][0])):
            lines.append("  - [ {: .6f},  {: .6f} ]".format(Phonon.ramanspectrum_data[0][0][j], 1e30*Phonon.ramanspectrum_data[dict[polarization]][1][j]))
    
    with open(Phonon.path+"Intensity.yaml", "w") as w:
        w.write("\n".join(lines))
    #
#

###############
# write IR Data
###############

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

####################################################################################################

##################
# read System Info
##################

def loadAtomsData(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)
    
    lattice = np.array(Data["Lattice"])
    points = Data["Points"]
    nat = len(points)
    symbols = [points[x]["Symbol"] for x in range(nat)]
    direct_positions = np.array([points[x]["Coordinates"] for x in range(nat)])
    cartesian_positions = [direct_positions[x][0]*lattice[0] + direct_positions[x][1]*lattice[1] + direct_positions[x][2]*lattice[2] for x in range(nat)]

    return symbols, cartesian_positions, lattice
#

def loadPhononsData(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)

    modes = Data["Modes"]
    freqs = [modes[x]["Frequency"] for x in range(len(modes))]
    eigenvecs = np.array([modes[x]["Eigenvector"] for x in range(len(modes))])
    modelist = np.array([modes[x]["Mode"] for x in range(len(modes))])

    return freqs, eigenvecs, modelist
#

def loadBornData(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)

    points = Data["Points"]
    nat = len(points)

    if "Born" in Data:
        born = Data["Born"]
    else:
        born = np.zeros((nat, 3, 3))
    
    return born
#

def loadEpsInfData(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)

    if "Dielectric_Tensor" in Data:
        eps_inf = Data["Dielectric_Tensor"]
    else:
        eps_inf = np.zeros((3,3))
    
    return eps_inf
#

def loadSymmetryData(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)

    if "Symmetry" in Data:
        symmetry = False
        pointgroup = ""
    else:
        symmetry = True
        pointgroup = Data["Pointgroup"]
    
    system = Data["System"]
    modes = Data["Modes"]
    labels = [modes[x]["Label"] for x in range(len(modes))]

    return system, symmetry, pointgroup, labels
#

def loadConstantRaman(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)

    keys = []
    values = []
    for mode in range(len(Data["Modes"])):
        keys.append(Data["Modes"][mode]["Mode"])
        values.append([Data["Modes"][mode]["Tensor"][0][0], 
                       Data["Modes"][mode]["Tensor"][1][1], 
                       Data["Modes"][mode]["Tensor"][2][2], 
                       Data["Modes"][mode]["Tensor"][0][1], 
                       Data["Modes"][mode]["Tensor"][1][2], 
                       Data["Modes"][mode]["Tensor"][0][2],
                       Data["Modes"][mode]["Tensor"][3]["perpendicular"],
                       Data["Modes"][mode]["Tensor"][4]["backscattering"]])
    #
    
    return dict(zip(keys, values))
#

def loadSpectrum(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)

    print(Data["Geometry"][0])

    ki = Data["Geometry"][0]["Polarization"].split("(")[0]
    ko = Data["Geometry"][0]["Polarization"].split(")")[-1]

    if ki == "x" and ko == "-x":
        qdir = (1,0,0)
    elif ki == "y" and ko == "-y":
        qdir = (0,1,0)
    elif ki == "z" and ko == "-z":
        qdir = (0,0,1)
    elif ki == "x" and ko == "y":
        qdir = (1,1,0)
    elif ki == "y" and ko == "z":
        qdir = (0,1,1)
    elif ki == "x" and ko == "z":
        qdir = (1,0,1)
    else:
        qdir = (0,0,0)
    #

    spectrum = 0
    return spectrum, qdir
#    
