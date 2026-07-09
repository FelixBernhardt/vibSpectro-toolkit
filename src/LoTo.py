#!/usr/bin/env python

#
# Lib to account for LO modes
#


import numpy as np
from parser.parserPhonopy import parsePhonopy
from spglib import get_symmetry_dataset
from Symmetries import periodTable, getIrrepsSymbols, getDegenerates

def normalize(v):
    return v / np.linalg.norm(v)
#

def collectMatrix(evec_dict, indices):
    # Build matrix with normalized eigenvectors as columns
    cols = [normalize(evec_dict[i].ravel()) for i in indices]
    return np.column_stack(cols)  # shape (3N, m)
#

# Build irrep subspaces from TO modes
def buildIrrepBases(E0_dict, irrep_label):
    # irrep_bases[Γ] = matrix whose columns span the Γ irrep subspace
    # irrep_groups[Γ] = list of TO indices belonging to Γ
    irrep_groups = {}
    for idx, Γ in irrep_label.items():
        irrep_groups.setdefault(Γ, []).append(idx)
    #
    irrep_bases = {}
    for Γ, idxs in irrep_groups.items():
        irrep_bases[Γ] = collectMatrix(E0_dict, idxs)
    #
    return irrep_bases
#

# Project NAC eigenvectors into each irrep
def projectIntoIrrep(EΓ, v):
    # proj_vec: projected vector in Γ-subspace
    # weight: total projection weight
    coeffs = EΓ.conj().T @ v
    proj_vec = EΓ @ coeffs
    weight = np.sum(np.abs(coeffs)**2)
    return proj_vec, weight
#

def decomposeNAC(ENAC_dict, irrep_bases):
    # nac_irrep_proj[j][Γ] = {
    #        "proj_vec": projected NAC eigenvector,
    #        "weight": projection weight }
    nac_irrep_proj = {}
    for j, v_raw in ENAC_dict.items():
        v = normalize(v_raw.ravel())
        nac_irrep_proj[j] = {}
        for Γ, EΓ in irrep_bases.items():
            proj_vec, weight = projectIntoIrrep(EΓ, v)
            nac_irrep_proj[j][Γ] = {
                "proj_vec": proj_vec,
                "weight": weight}
        #
    #
    return nac_irrep_proj
#

# orthogonalization inside each irrep
def alignSubspace(E_block, groupsize, qdir):
    nat3 = len(E_block[:,0])
    nmodes = len(E_block[0,:])
    E_rot = np.zeros_like(E_block)
    q = np.tile(qdir, int(nat3/3))
    mode = 0
    if groupsize == 1:
        return E_block
    elif groupsize == 2: 
        for j in range(int(nmodes/2)):
            v = E_block[:,mode:mode+1] @ (E_block[:,mode:mode+1].conj().T @ q)

            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                print("[alignSubspace]: Direction q has negligible projection onto TO E subspace, rotations not applied.")
                return E_block

            e_parallel = v / norm_v
            
            # Build orthogonal partner inside the same subspace
            # Start from one of the original TO modes and Gram-Schmidt
            w = E_block[:,mode] - np.vdot(e_parallel, E_block[:,mode]) * e_parallel
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-12:
                # If unlucky, use the other TO mode
                w = E_block[:,mode+1] - np.vdot(e_parallel, E_block[:,mode+1]) * e_parallel
                norm_w = np.linalg.norm(w)
                if norm_w < 1e-12:
                    print("[alignSubspace]: Direction q has negligible projection onto TO E subspace, rotations not applied.")
                    return E_block

            e_perp = w / norm_w

            E_rot[:,mode] = e_parallel
            E_rot[:,mode+1] = e_perp
            mode += 2
        #
    elif groupsize == 3: 
        for j in range(int(nmodes/2)):
            v = E_block[:,mode:mode+1] @ (E_block[:,mode:mode+1].conj().T @ q)

            norm_v = np.linalg.norm(v)
            if norm_v < 1e-12:
                print("[alignSubspace]: Direction q has negligible projection onto TO E subspace, rotations not applied.")
                return E_block
            #
            e_parallel = v / norm_v
            
            # Build orthogonal partner inside the same subspace
            # Start from one of the original TO modes and Gram-Schmidt
            w = E_block[:,mode] - np.vdot(e_parallel, E_block[:,mode]) * e_parallel
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-12:
                # If unlucky, use the other TO mode
                w = E_block[:,mode+1] - np.vdot(e_parallel, E_block[:,mode+1]) * e_parallel
                norm_w = np.linalg.norm(w)
                if norm_w < 1e-12:
                    # If unlucky, use the other TO mode
                    w = E_block[:,mode+2] - np.vdot(e_parallel, E_block[:,mode+2]) * e_parallel
                    norm_w = np.linalg.norm(w)
                    if norm_w < 1e-12:
                        print("[alignSubspace]: Direction q has negligible projection onto TO E subspace, rotations not applied.")
                        return E_block
                    #
                #
            #
            e_perp1 = w / norm_w

            # Build second orthogonal partner inside the same subspace
            # Start from one of the original TO modes and Gram-Schmidt
            w = E_block[:,mode] - np.vdot(e_parallel, E_block[:,mode]) * e_parallel - np.vdot(e_perp1, E_block[:,mode]) * e_perp1
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-12:
                # If unlucky, use the other TO mode
                w = E_block[:,mode+1] - np.vdot(e_parallel, E_block[:,mode+1]) * e_parallel - np.vdot(e_perp1, E_block[:,mode+1]) * e_perp1
                norm_w = np.linalg.norm(w)
                if norm_w < 1e-12:
                    # If unlucky, use the other TO mode
                    w = E_block[:,mode+2] - np.vdot(e_parallel, E_block[:,mode+2]) * e_parallel - np.vdot(e_perp1, E_block[:,mode+2]) * e_perp1
                    norm_w = np.linalg.norm(w)
                    if norm_w < 1e-12:
                        print("[alignSubspace]: Direction q has negligible projection onto TO E subspace, rotations not applied.")
                        return E_block
                    #
                #
            #
            e_perp2 = w / norm_w

            E_rot[:,mode] = e_parallel
            E_rot[:,mode+1] = e_perp1
            E_rot[:,mode+2] = e_perp2
            mode += 3
        #
    #
    else:
        print("[alignSubspace]: Invalid degenerate group detected, rotations not applied.")
        return E_block
    #
    return E_rot
#

def assignLabel(ENAC_dict, nac_irrep_proj, irrep_label):
    nmodes = len(ENAC_dict)
    possible_labels = set(irrep_label.values())
    # assign all modes to their best match
    irrep_label_nac = {}
    for mode in range(1,nmodes+1):
        weight = []
        dict = {}
        for label in possible_labels:
            weight.append(nac_irrep_proj[mode][label]["weight"])
            dict[weight[-1]] = label
        #
        irrep_label_nac[mode] = dict[np.max(weight)]
        #
    #

    return irrep_label_nac
#

def LOTOassign(E0_dict, ENAC_dict, degenerate_groups, irrep_label, qdir):
    
    for mode in range(1,len(E0_dict)+1):
        if any(mode in x for x in degenerate_groups):
            continue
        else:
            degenerate_groups.append([mode])
        #
    #

    irrep_bases = buildIrrepBases(E0_dict, irrep_label)
    nac_irrep_proj = decomposeNAC(ENAC_dict, irrep_bases)

    # the nac mode labels
    irrep_label_nac = assignLabel(ENAC_dict, nac_irrep_proj, irrep_label)

    results = {}

    # Process each degenerate TO block (each block belongs to one irrep)
    for group in degenerate_groups:
        # Determine irrep of this block
        label = irrep_label[group[0]]
        chosen = [mode for mode in range(1,len(ENAC_dict)+1) if irrep_label_nac[mode] == label]

        # Build TO subspace matrix for this block
        E0_block = collectMatrix(E0_dict, group)

        # Build NAC block matrix from projected vectors
        ENAC_block = np.column_stack( [normalize(nac_irrep_proj[mode][label]["proj_vec"]) for mode in chosen] )

        # Align subspaces
        ENAC_rot = alignSubspace(ENAC_block, len(group), qdir)
        E0_rot = alignSubspace(E0_block, len(group), qdir)

        # find best match
        for k, indx0 in enumerate(group):
            weight = []
            dict = {}
            for j, indxNAC in enumerate(chosen):
                #e0 = E0_block[:, k]
                #eN = ENAC_block[:, j]
                e0 = E0_rot[:, k]
                eN = ENAC_rot[:, j]
                weight.append( np.abs(np.vdot(e0, eN))**2 )
                dict[weight[-1]] = indxNAC
            #
            results[indx0] = dict[np.max(weight)]
        #            
    #

    return results
#

def getLOFreqs(path, modelist, qdir_cart, qdir_direct, ordering, eigvecs, degenerates, labels):
    # get the LO modes corresponding to the direction to be analyzed
    #<phonopy --readfc --sym-fc --writedm --qpoints="0 0 0" --nac --q-direction="0 0 1">

    # read the LO modes
    print("[getLOFreqs]: Using cartesian q-direction "+str(qdir_cart))
    eigvals_tmp, eigvecs_tmp, _, _, basis, nat, elements, cPos, _ = parsePhonopy(path, [qdir_cart, qdir_direct])

    # phonopy always provides all modes in ascending order
    eigvecs_pt = dict(zip([j for j in range(1,3*nat+1)], eigvecs_tmp))
    eigvals_pt = dict(zip([j for j in range(1,3*nat+1)], eigvals_tmp))

    if len(modelist) != 3*nat-3:
        # read all the TO modes from phonopy
        eigvals, eigvecs, _, _, basis, nat, elements, cPos, _ = parsePhonopy(path, None)
        direct = np.dot(cPos, np.linalg.inv(basis))
        eigvecs = dict(zip([j for j in range(1,3*nat+1)], eigvecs))
        eigvals = dict(zip([j for j in range(1,3*nat+1)], eigvals))

        _dataset = get_symmetry_dataset((basis, direct, [periodTable[element] for element in elements]), symprec=1.e-5)
        pointgroup = str(_dataset["pointgroup"])
        _labels_tmp = getIrrepsSymbols(path, basis, direct, elements, pointgroup)
        labels = dict(enumerate([_labels_tmp[i-1] for i in range(1,3*nat+1)], start=1))
        degenerates = getDegenerates(range(1,3*nat+1), eigvals, labels, prec=1e0)
    #

    # match the TO to the LO modes
    LoToDict = LOTOassign(eigvecs, eigvecs_pt, degenerates, labels, qdir_cart)
    
    # reorder and assign the frequencies
    eigvalsLO = {}
    if ordering == "ascending":
        for mode in modelist:
            eigvalsLO[mode] = eigvals_pt[LoToDict[mode]]
        #
    elif ordering == "descending" and len(modelist) != 3*nat-3:
        for mode in modelist:
            eigvalsLO[mode] = eigvals_pt[LoToDict[3*nat+1-mode]]
        #
    elif ordering == "descending" :
        for mode in modelist:
            eigvalsLO[mode] = eigvals_pt[LoToDict[mode]]
        #
    #

    return eigvalsLO
#