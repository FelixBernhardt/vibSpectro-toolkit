import numpy as np
from src.Symmetries import portoq, findInListOfList
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
        lines.append("Source:")
        if mode in [x[0] for x in Phonon.degenerates] or Phonon.nosym == True:
            lines.append("- " + Phonon.path + "displacements/mode" + str(mode) + "_1/" + outfile)
            lines.append("- " + Phonon.path + "displacements/mode" + str(mode) + "_-1/" + outfile)
        else:
            idx = findInListOfList(Phonon.degenerates, mode)
            lines.append("- " + Phonon.path + "displacements/mode" + str(Phonon.degenerates[idx[0]][0]) + "_1/" + outfile)
            lines.append("- " + Phonon.path + "displacements/mode" + str(Phonon.degenerates[idx[0]][0]) + "_-1/" + outfile)
        lines.append("Units:")
        lines.append("  Phonon Wavelength: cm⁻1")
        lines.append("  Laser_Frequency: eV")
        lines.append("  Raman_Tensor: 10⁻30 Cm^2/V")
        lines.append("Mode:")
        lines.append("- Index: " + str(mode))
        lines.append("  Label: " + Phonon.labels[mode])
        lines.append("  Wavelength: {: .6f}".format(Phonon.eigenfreqs[mode]))
        lines.append("  Raman_Tensor:")
        
        for i in range(len(Phonon.ramantensors_data[mode])):
            lines.append("  - Laser_Frequency: {: .6f}".format(Phonon.ramantensors_data[mode][i][0].real))
            lines.append("    perpendicular : {: .6f}".format(Phonon.ramantensors_data[mode][i][7].real))
            lines.append("    backscattering: {: .6f}".format(Phonon.ramantensors_data[mode][i][8].real))
            lines.append("    Raman_Tensor:")
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[mode][i][1], Phonon.ramantensors_data[mode][i][4], Phonon.ramantensors_data[mode][i][6]))
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[mode][i][4], Phonon.ramantensors_data[mode][i][2], Phonon.ramantensors_data[mode][i][5]))
            lines.append("    - [ {: .6f},  {: .6f},  {: .6f} ]".format(Phonon.ramantensors_data[mode][i][6], Phonon.ramantensors_data[mode][i][5], Phonon.ramantensors_data[mode][i][3]))
        
        with open(Phonon.path+"Ramantensors/alpha"+str(mode)+".yaml", "w") as w:
            w.write("\n".join(lines))
        #
    #
#

# Raman tensors at constant excitation energy
def writeConstantRaman(Phonon):
    if Phonon.qdir_cartesian == (1,0,0):
        ki = "x"
        ko = "-x"
    elif Phonon.qdir_cartesian == (0,1,0):
        ki = "y"
        ko = "-y"
    elif Phonon.qdir_cartesian == (0,0,1):
        ki = "z"
        ko = "-z"
    elif Phonon.qdir_cartesian == (1,1,0):
        ki = "x"
        ko = "y"
    elif Phonon.qdir_cartesian == (0,1,1):
        ki = "y"
        ko = "z"
    elif Phonon.qdir_cartesian == (1,0,1):
        ki = "x"
        ko = "z"
    else:
        ki = "x"
        ko = "x"
    #

    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Source: " + Phonon.path+"Ramantensors/alpha*.yaml")
    lines.append("Units:")
    lines.append(" Frequency: cm⁻1")
    lines.append(" Laser_Frequency: eV")
    lines.append(" Raman_Tensor: 10⁻30 Cm^2/V")
    lines.append("Laser_frequency: {: .6f}".format(Phonon.photon_freq))
    lines.append("Geometry: "+ki+"(..)"+ko)
    lines.append("Modes:")
    for mode in Phonon.Ramanmodelist:
        lines.append("- Mode: " + str(mode))
        lines.append("  Label: \""+ Phonon.labels[mode] +"\"")
        if Phonon.LOcorr == True:
            lines.append("  Frequency: {: .6f}".format(Phonon.eigenfreqs_LO[mode]))
        else:
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
    if Phonon.qdir_cartesian == (1,0,0):
        ki = "x"
        ko = "-x"
    elif Phonon.qdir_cartesian == (0,1,0):
        ki = "y"
        ko = "-y"
    elif Phonon.qdir_cartesian == (0,0,1):
        ki = "z"
        ko = "-z"
    elif Phonon.qdir_cartesian == (1,1,0):
        ki = "x"
        ko = "y"
    elif Phonon.qdir_cartesian == (0,1,1):
        ki = "y"
        ko = "z"
    elif Phonon.qdir_cartesian == (1,0,1):
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
        lines.append("        " + Phonon.path + "qpoints_"+portoq[Phonon.qdir_cartesian] + ".yaml")
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
    lines.append(" Frequency: cm⁻1")

    lines.append("Source:")
    lines.append("- "+Phonon.path+Phonon.file)
    if Phonon.file == "phonopy.yaml":
        lines.append("- "+Phonon.path+"qpoints.yaml")
        lines.append("- "+Phonon.path+"BORN")

    lines.append("Geometry:")
    for polarization in ["x", "y", "z"]:
        lines.append("- Polarization: E || "  + polarization )
        lines.append("  Spectrum:")
        for j in range(len(Phonon.IR_data[0])):
            lines.append("  - [ {: .6f},  {: .6f} ]".format(Phonon.IR_data[0][j].real, Phonon.IR_data[dict[polarization]][j]))
    
    with open(Phonon.path+"IR.yaml", "w") as w:
        w.write("\n".join(lines))
    #
#

def writeReflectanceSpectrum(Phonon):
    dict = {"x": 1, "y": 2, "z": 3}
    lines = []
    lines.append("System: " + Phonon.name)
    lines.append("Units:")
    lines.append(" Frequency: cm⁻1")

    lines.append("Source:")
    lines.append("- "+Phonon.path+Phonon.file)
    if Phonon.file == "phonopy.yaml":
        lines.append("- "+Phonon.path+"qpoints.yaml")
        lines.append("- "+Phonon.path+"BORN")

    lines.append("Geometry:")
    for polarization in ["x", "y", "z"]:
        lines.append("- Polarization: E || "  + polarization )
        lines.append("  Spectrum:")
        for j in range(len(Phonon.reflectance_data[0])):
            lines.append("  - [ {: .6f},  {: .6f} ]".format(Phonon.reflectance_data[0][j], Phonon.reflectance_data[dict[polarization]][j]))
    
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

############################
# Raman tensors and spectrum
############################
def loadRamanTensor(path, mode):
    # load a single, frequency dependent ramantensor
    filename = path+"Ramantensors/alpha"+str(mode)+".yaml"
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)

    tensor = np.zeros((len(Data["Mode"][0]["Raman_Tensor"]), 9), dtype=complex)
    for i in range(len(Data["Mode"][0]["Raman_Tensor"])):
        tensor[i, 0] = Data["Mode"][0]["Raman_Tensor"][i]["Laser_Frequency"]
        tensor[i, 1] = Data["Mode"][0]["Raman_Tensor"][i]["Raman_Tensor"][0][0]
        tensor[i, 2] = Data["Mode"][0]["Raman_Tensor"][i]["Raman_Tensor"][1][1]
        tensor[i, 3] = Data["Mode"][0]["Raman_Tensor"][i]["Raman_Tensor"][2][2]
        tensor[i, 4] = Data["Mode"][0]["Raman_Tensor"][i]["Raman_Tensor"][0][1]
        tensor[i, 5] = Data["Mode"][0]["Raman_Tensor"][i]["Raman_Tensor"][1][2]
        tensor[i, 6] = Data["Mode"][0]["Raman_Tensor"][i]["Raman_Tensor"][0][2]
        tensor[i, 7] = Data["Mode"][0]["Raman_Tensor"][i]["perpendicular"]
        tensor[i, 8] = Data["Mode"][0]["Raman_Tensor"][i]["backscattering"]
    
    return tensor    
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
        qdir = (1,0,0)
    #

    spectrum = np.zeros((8, 2, len(Data["Geometry"][0]["Spectrum"])))

    for geometry in range(len(Data["Geometry"])):
        if Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "xx":
            col = 0
        elif Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "yy":
            col = 1
        elif Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "zz":
            col = 2
        elif Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "xy" or Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "yx":
            col = 3
        elif Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "yz" or Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "zy":
            col = 4
        elif Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "xz" or Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "zx":
            col = 5
        elif Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "perpendicular":
            col = 6
        elif Data["Geometry"][geometry]["Polarization"].split("(")[-1].split(")")[0] == "backscattering":
            col = 7
        else:
            print("[loadSpectrum]: Invalid polarization found in "+filename+". Denoting it with xx and continuing...")
            col = 0
        
        for i in range(len(Data["Geometry"][geometry]["Spectrum"])):
            spectrum[col, 0, i] = Data["Geometry"][geometry]["Spectrum"][i][0]
            spectrum[col, 1, i] = Data["Geometry"][geometry]["Spectrum"][i][1]*10**(-30)

    return spectrum, qdir
#    

#############################
# IR spectrum and reflectance
#############################

def loadIR(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)
    
    spectrum = np.zeros((4, len(Data["Geometry"][0]["Spectrum"])), dtype=complex)

    for i in range(len(Data["Geometry"][0]["Spectrum"])):
        spectrum[0, i] = Data["Geometry"][0]["Spectrum"][i][0]
        spectrum[1, i] = Data["Geometry"][0]["Spectrum"][i][1]
        spectrum[2, i] = Data["Geometry"][1]["Spectrum"][i][1]
        spectrum[3, i] = Data["Geometry"][2]["Spectrum"][i][1]
                   
    return spectrum
#

def loadReflectance(filename):
    with open(filename, "r") as f:
        Data = yaml.safe_load(f)
    
    spectrum = np.zeros((4, len(Data["Geometry"][0]["Spectrum"])))

    for i in range(len(Data["Geometry"][0]["Spectrum"])):
        spectrum[0, i] = Data["Geometry"][0]["Spectrum"][i][0]
        spectrum[1, i] = Data["Geometry"][0]["Spectrum"][i][1]
        spectrum[2, i] = Data["Geometry"][1]["Spectrum"][i][1]
        spectrum[3, i] = Data["Geometry"][2]["Spectrum"][i][1]
                   
    return spectrum
#