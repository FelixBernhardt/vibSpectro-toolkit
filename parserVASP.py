#!/usr/bin/env python

#
# VASP parsers
#

import re
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Dict

def T(m):
    p = [[ m[i][j] for i in range(len( m[j] )) ] for j in range(len( m )) ]
    return p
#

def flatten(t):
    a = []
    for sublist in t:
        if isinstance(sublist, str):
            a.append(sublist)
        else:
            for item in sublist:
                a.append(item)
            #
        #
    #
    return a
#

def _parse_array_block(block: ET.Element) -> List[Dict[str, float]]:
    """
    Parse a dielectricfunction <imag> or <real> block with VASP's array/set/r layout.

    Returns a list of dicts, one per row, keyed by the <field> names.
    Example keys: ['energy', 'xx', 'yy', 'zz', 'xy', 'yz', 'zx']
    """
    arr = block.find("array")
    if arr is None:
        return []
    #

    # Collect field names in order (skip <dimension>, only take <field>)
    fields = [f.text.strip() for f in arr.findall("field")]
    if not fields:
        # Some versions have <v name="..."> instead; try fallback
        fields_nodes = arr.findall(".//field")
        fields = [f.text.strip() for f in fields_nodes]
    #

    # Collect rows under one or more <set> elements
    rows = []
    for set_el in arr.findall("set"):
        for r_el in set_el.findall("r"):
            # Split by whitespace
            parts = r_el.text.strip().split()
            if len(parts) != len(fields):
                # Warning if the row length doesnt match fields 
                print("[getOpticsVASP]: inconsistency in vasprun.xml found, continuing anyway... ")
                continue
            #
            values = [float(p) for p in parts]
            row = {field: val for field, val in zip(fields, values)}
            rows.append(row)
        #
    #
    return rows
#

def getOpticsVASP(file):
    tree = ET.parse(file)
    root = tree.getroot()

    tree.getroot()

    result = {"real": [], "imag": []}

    for df in root.findall(".//dielectricfunction"):
        imag = df.find("imag")
        if imag is not None:
            result["imag"].extend(_parse_array_block(imag))
        #

        real = df.find("real")
        if real is not None:
            result["real"].extend(_parse_array_block(real))
        #
    #

    # formatting
    leng = len(result["real"])
    w = np.array([result["real"][j]["energy"] for j in range(leng)], dtype=float)
    Im = np.array([[result["imag"][j]["xx"] for j in range(leng)],
                   [result["imag"][j]["yy"] for j in range(leng)], 
                   [result["imag"][j]["zz"] for j in range(leng)],
                   [result["imag"][j]["xy"] for j in range(leng)],
                   [result["imag"][j]["yz"] for j in range(leng)],
                   [result["imag"][j]["zx"] for j in range(leng)] ], dtype=float)

    
    Re = np.array([[result["real"][j]["xx"] for j in range(leng)],
                   [result["real"][j]["yy"] for j in range(leng)], 
                   [result["real"][j]["zz"] for j in range(leng)],
                   [result["real"][j]["xy"] for j in range(leng)],
                   [result["real"][j]["yz"] for j in range(leng)],
                   [result["real"][j]["zx"] for j in range(leng)] ], dtype=float)

    return w, Im, Re
#

def getBornVASP(file, nat):
    try: 
        outcar_fh = open(file, "r")
    except IOError:
        print("[getBornVASP]: ERROR Couldn't open "+file+"\n")
        return np.zeros((nat, 3, 3))
    #

    outcar_fh.seek(0)
    while True:
        line = outcar_fh.readline()
        if not line:
            break
        #
        if "BORN EFFECTIVE CHARGES (in e, cummulative output)" in line or \
           "BORN EFFECTIVE CHARGES (including local field effects) (in |e|, cummulative output)" in line:
            born = np.zeros((nat,3,3))
            outcar_fh.readline() # ----------------------------------------------------
            #
            for i in range(nat):
                outcar_fh.readline() # ion X
                for j in range(3):
                    line = outcar_fh.readline().split()
                    born[i,j] = [float(line[1]), float(line[2]), float(line[3])]
                #
            #
            #format: born[ION][COLUMN][LINE]
            # check for charge neutrality (just in case...)
            tot = np.zeros((3))
            for alpha in range(3):
                for i in range(nat):
                    for beta in range(3):
                        tot[alpha] += born[i][alpha][beta]
                    #
                #
                if tot[alpha] > 1.e-3:
                    print("[getBornVASP]: WARNING The charge neutrality condition for direction " + str(alpha) + " is not fullfilled")
                #
            #
            return born
        #
    #
    print("[getBornVASP]: ERROR Couldn't find 'BORN EFFECTIVE CHARGES' in OUTCAR. Exiting...")
    return np.zeros((nat, 3, 3))
#

def getEpsInfVASP(file):
    try: 
        eps = []
        with open(outcar) as f:
            lines = f.readlines()
        #
        start = None
        for i, line in enumerate(lines):
            if "MACROSCOPIC STATIC DIELECTRIC TENSOR" in line:
                start = i + 2
                break
        #
        if start is None:
            return np.zeros((3, 3))
        #
        for i in range(start, start + 3):
            row = list(map(float, lines[i].split()))
            eps.append(row)
        #
        return np.array(eps)
    
    except IOError:
        print("[getEpsInfVASP]: ERROR Couldn't open "+file+"\n")
        return np.zeros((3, 3))
    #
#

def ModeParserVASP(outcar_fh, modelist, nat, case):
    eigvals = np.zeros(3*nat)
    eigvecs = np.zeros((3*nat, nat, 3))
    norms   = np.zeros(3*nat)
    if case == 1:
        outcar_fh.readline() # empty line
        outcar_fh.readline() # Eigenvectors and eigenvalues of the dynamical matrix
        outcar_fh.readline() # ----------------------------------------------------
        outcar_fh.readline() # empty line
    elif case == 2:
        outcar_fh.readline() # ----------------------------------------------------
        outcar_fh.readline() # empty line
    else:
        print("[ModeParserVASP]: Invalid case specified, exiting...")
        return [0], np.zeros(3), [0]
    #
    for i in range(np.max(modelist)):
        outcar_fh.readline() # empty line
        p = re.search(r'^\s*(\d+).+?([\.\d]+) cm-1', outcar_fh.readline())
        eigvals[i] = float(p.group(2))
        # look for imaginary modes
        if p.group(0)[7] == 'i':
            eigvals[i] = -eigvals[i]
        #
        outcar_fh.readline() # X         Y         Z           dx          dy          dz
        eigvec = []
        #
        for j in range(nat):
            tmp = outcar_fh.readline().split()
            # stupid VASP spacing bug...
            if len(tmp) < 6:
                tmp3 = []
                for s in range(len(tmp)):
                    if "-" in tmp[s]:
                        tmp2 = tmp[s].split("-")
                        if len(tmp2) == 2:
                            tmp2[1] = "-"+tmp2[1]
                        elif len(tmp2) == 3:
                            tmp2[1] = "-"+tmp2[1]
                            tmp2[2] = "-"+tmp2[2]
                    else:
                        tmp2 = tmp[s]
                    #
                    tmp3.append(tmp2)
                #
                tmp = flatten(tmp3)
                tmp = [x for x in tmp if x != ""] 
            #
            #tmp = re.split("[\s-]+", outcar_fh.readline())
            #
            eigvec.append([ float(tmp[x]) for x in range(3,6) ])
            #
        eigvecs[i] = np.array(eigvec)
        norms[i] = np.sqrt( sum( [abs(x)**2 for sublist in eigvec for x in sublist] ) )
    #    
    return eigvals, eigvecs, norms
#

def getModesVASP(file, modelist, nat):
    try: 
        outcar_fh = open(file, "r")
    except IOError:
        print("[getModesVASP]: ERROR Couldn't open OUTCAR\n")
        return np.zeros(3)
    #
    outcar_fh.seek(0)
    while True:
        line = outcar_fh.readline()
        if not line:
            break
        #
        if "Eigenvectors after division by SQRT(mass)" in line:
            eigvals, eigvecs, norms = ModeParserVASP(outcar_fh, modelist, nat, 1)
            return eigvals, eigvecs, norms
        elif "Eigenvectors and eigenvalues of the dynamical matrix" in line:
            eigvals, eigvecs, norms = ModeParserVASP(outcar_fh, modelist, nat, 2)
            return eigvals, eigvecs, norms
        #
    #
    print("[getModesVASP]: ERROR Couldn't find 'Eigenvectors and eigenvalues of the dynamical matrix' in OUTCAR")
    return np.zeros(3)
#

def MAT_m_VEC(m, v):
    p = [ 0.0 for i in range(len(v)) ]
    for i in range(len(m)):
        assert len(v) == len(m[i]), "[Mat_m_VEC]: Length of the matrix row is not equal to the length of the vector"
        p[i] = sum( [ m[i][j]*v[j] for j in range(len(v)) ] )
    return p
#

def getCellVASP(file):
    try: 
        poscar_fh = open(file, "r")
    except IOError:
        print("[getCellVASP]: ERROR Couldn't open POSCAR\n")
        return 1, np.eye(3), np.zeros(3), [""]
    #
    poscar_fh.seek(0) # just in case
    lines = poscar_fh.readlines()
    scale = float(lines[1])
    if scale < 0.0:
        vol = -scale
        cell = np.zeros((3, 3))
        norm = np.zeros((3))
        for i in range(3):
            # Get the first three numbers from the line
            cell[i, :] = [float(x) for x in lines[i+2].split()[0:3]]
        #
        # set lattice constant, reset it later
        alat = np.power(vol/np.linalg.det(cell), 1./3)
        for i in range(3):
            # Get the first three numbers from the line
            cell[i, :] = [float(x) * alat for x in lines[i+2].split()[0:3]]
            norm[i] = np.linalg.norm(cell[i, :])
        #      
        b = cell
    #
    else:
        b = []
        for i in range(2, 5):
            b.append([float(x)*scale for x in lines[i].split()[:3]])
        #
        vol = b[0][0]*b[1][1]*b[2][2] + b[1][0]*b[2][1]*b[0][2] + b[2][0]*b[0][1]*b[1][2] - \
              b[0][2]*b[1][1]*b[2][0] - b[2][1]*b[1][2]*b[0][0] - b[2][2]*b[0][1]*b[1][0]
        #
    #
    try:
        num_atoms = [int(x) for x in lines[5].split()]
        line_at = 6
    except ValueError:
        num_atoms = [int(x) for x in lines[6].split()]
        line_at = 7
    atom_types = [str(x) for x in lines[line_at-2].split()]
    nat = sum(num_atoms)
    #
    seldyn = 0
    if lines[line_at][0].lower() == 's':
        seldyn = 1
        line_at += 1
    #
    if (lines[line_at][0].lower() == 'c' or lines[line_at][0].lower() == 'k'):
        is_scaled = False
    else:
        is_scaled = True
    #
    line_at += 1
    #
    positions = []
    for i in range(line_at, line_at + nat):
        pos = [float(x) for x in lines[i].split()[:3]]
        #
        if is_scaled:
            pos = MAT_m_VEC(T(b), pos)
        #
        positions.append(pos)
    #
    # add element symbols
    indices = []
    counter = 0
    counter2 = 0
    for j in range(nat):
        if int(counter) == int(num_atoms[counter2]):
            counter2 += 1
            counter = 0
        #
        indices.append(atom_types[counter2])
        counter += 1
    #
    return nat, np.array(b), np.array(positions), indices
#

def writePOSCAR(nat, basis, positions, elements, file, mode, disp, stepsize, eigvec, norm):
    poscar_fh = open(file+"/POSCAR", "w")
    poscar_fh.write("generated by RamanPy, mode "+str(mode)+" displacement "+str(disp*stepsize)+"\n")
    
    # basis
    poscar_fh.write("1.0\n")
    for i in range(3):
        cell = [ basis[i][l] for l in range(3)]
        poscar_fh.write( '%15.10f %15.10f %15.10f\n' % (cell[0], cell[1], cell[2]) )
    #

    # calculate elements line
    # count occurence of same elements in a row
    counter = 1
    counter_list = []
    element_list = []
    element_list.append( elements[0] )
    for j in range(nat - 1):
        j += 1
        if elements[j][0] == elements[j-1][0]:
            counter += 1
        else:
            counter_list.append(counter)
            counter = 1
            element_list.append( elements[j] )
        #
    #
    counter_list.append(counter)
    elementstr = ""
    counterstr = ""
    for i in range(len(element_list)):
        elementstr += " "+element_list[i]
        counterstr += " "+str(counter_list[i])
    poscar_fh.write(elementstr+"\n")
    poscar_fh.write(counterstr+"\n")

    # positions
    poscar_fh.write("Cartesian\n")
    for i in range(nat):
        pos_disp = [ positions[i][l] + eigvec[i][l]*stepsize*disp/norm for l in range(3)]
        poscar_fh.write( '%15.10f %15.10f %15.10f\n' % (pos_disp[0], pos_disp[1], pos_disp[2]) )
    #
    poscar_fh.close()
#

def linkVASP(file):
    import os
    os.chdir(file)
    os.system("ln -s ../../KPOINTS KPOINTS")
    os.system("ln -s ../../POTCAR POTCAR")
    os.system("ln -s ../../INCAR INCAR")
    os.chdir("../..")
#