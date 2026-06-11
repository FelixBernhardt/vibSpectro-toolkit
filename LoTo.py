#!/usr/bin/env python

#
# Lib to account for LO modes
#


import numpy as np
from parserPhonopy import parsePhonopy
from Symmetries import flatten, e_charge
from calcTensors import placzeckInvs

# ------------------------------
# Basic utilities
# ------------------------------

def normalize(v):
    return v / np.linalg.norm(v)

def collect_matrix(evec_dict, indices):
    """Build matrix with normalized eigenvectors as columns."""
    cols = [normalize(evec_dict[i].ravel()) for i in indices]
    return np.column_stack(cols)  # shape (3N, m)

# ------------------------------
# Build irrep subspaces from TO modes
# ------------------------------

def build_irrep_bases(E0_dict, irrep_label):
    """
    Returns:
        irrep_bases[Γ] = matrix whose columns span the Γ irrep subspace
        irrep_groups[Γ] = list of TO indices belonging to Γ
    """
    irrep_groups = {}
    for idx, Γ in irrep_label.items():
        irrep_groups.setdefault(Γ, []).append(idx)

    irrep_bases = {}
    for Γ, idxs in irrep_groups.items():
        irrep_bases[Γ] = collect_matrix(E0_dict, idxs)

    return irrep_bases, irrep_groups

# ------------------------------
# Project NAC eigenvectors into each irrep
# ------------------------------

def project_into_irrep(EΓ, v):
    """
    Project vector v into the irrep subspace spanned by EΓ.
    Returns:
        proj_vec: projected vector in Γ-subspace
        weight: total projection weight
    """
    coeffs = EΓ.conj().T @ v
    proj_vec = EΓ @ coeffs
    weight = np.sum(np.abs(coeffs)**2)
    return proj_vec, weight

def decompose_nac_into_irreps(ENAC_dict, irrep_bases):
    """
    Returns:
        nac_irrep_proj[j][Γ] = {
            "proj_vec": projected NAC eigenvector,
            "weight": projection weight
        }
    """
    nac_irrep_proj = {}
    for j, v_raw in ENAC_dict.items():
        v = normalize(v_raw.ravel())
        nac_irrep_proj[j] = {}
        for Γ, EΓ in irrep_bases.items():
            proj_vec, weight = project_into_irrep(EΓ, v)
            nac_irrep_proj[j][Γ] = {
                "proj_vec": proj_vec,
                "weight": weight
            }
    return nac_irrep_proj

# ------------------------------
# SVD alignment inside each irrep
# ------------------------------

def align_subspace(E0_block, ENAC_block):
    """
    Perform SVD alignment:
        S = E0^† ENAC
        S = U Σ V^†
        ENAC_rot = ENAC V
    """
    S = E0_block.conj().T @ ENAC_block
    U, Sigma, Vh = np.linalg.svd(S)
    ENAC_rot = ENAC_block @ Vh.conj().T
    return ENAC_rot, Sigma

def longitudinal_character(evec, qhat, qhat1, qhat2):
    # evec shape: (natoms, 3)
    flat = evec.reshape(-1, 3)
    projLO = np.sum(flat @ qhat)
    projTO = np.sum(flat @ qhat1) + np.sum(flat @ qhat2)
    print(projLO-projTO)
    if projLO - projTO > 0.4:
        return "LO"
    else:
        return "TO"
#

def rec_assign_label(E0_dict, ENAC_dict, nac_irrep_proj, irrep_label):
    nmodes = len(ENAC_dict)
    qdir = [0,0,1]
    qperp1 = [1,0,0]
    qperp2 = [0,1,0]
    possible_labels = ["A1", "A2", "E"]
    # first assign all modes to their best match
    irrep_label_nac = {}
    for mode in range(1,nmodes+1):
        weight = []
        dict = {}
        for label in possible_labels:
            weight.append(nac_irrep_proj[mode][label]["weight"])
            dict[weight[-1]] = label
        #
        if np.max(weight) > 0.99:
            irrep_label_nac[mode] = dict[np.max(weight)]
        else:
            #weight.remove(np.max(weight))
            irrep_label_nac[mode] = dict[np.max(weight)]
        #
    #

    # look which modes are LO and which are TO
    LOs = {}
    TOs = {}
    for mode in range(1,nmodes+1):
        weight = longitudinal_character(ENAC_dict[mode], qdir, qperp1, qperp2)
        print(mode, weight)

    # check if the best matches are consistent
    # if not, start with the worst match and reassign the mode
    nlabels = {}
    nlabels_nac = {}
    for label in possible_labels:
        counter = 0
        counter_nac = 0
        for mode in range(1,nmodes+1):
            if irrep_label[mode] == label:
                counter += 1
            if irrep_label_nac[mode] == label:
                counter_nac += 1
            #
        #
        nlabels[label] = counter
        nlabels_nac[label] = counter_nac
    #
    print(nlabels)
    print(nlabels_nac)
    #    

    return irrep_label_nac
#

# ------------------------------
# Main routine: irrep-clean LO/TO assignment
# ------------------------------

def LOTOassign(E0_dict, ENAC_dict, degenerate_groups, irrep_label):
    
    for mode in range(1,len(E0_dict)+1):
        if any(mode in x for x in degenerate_groups):
            continue
        else:
            degenerate_groups.append([mode])
        #
    #

    irrep_bases, irrep_groups = build_irrep_bases(E0_dict, irrep_label)
    nac_irrep_proj = decompose_nac_into_irreps(ENAC_dict, irrep_bases)

    
    print(irrep_label)

    # recursively assign the mode labels to the OVERALL best match
    irrep_label_nac = rec_assign_label(E0_dict, ENAC_dict, nac_irrep_proj, irrep_label)
    print(irrep_label_nac)

    results = {}

    # Process each degenerate TO block (each block belongs to one irrep)
    for group in degenerate_groups:
        # Determine irrep of this block
        Γ = irrep_label[group[0]]
        EΓ = irrep_bases[Γ]

        #print(Γ)

        # Build TO subspace matrix for this block
        E0_block = collect_matrix(E0_dict, group)
        m = len(group)

        # Select NAC modes with largest projection weight into this irrep
        weights = []
        for j in ENAC_dict.keys():
            w = nac_irrep_proj[j][Γ]["weight"]
            #print(j)
            weights.append((w, j))
        weights.sort(reverse=True)
        chosen = [j for (_, j) in weights[:m]]

        # Build NAC block matrix from projected vectors
        ENAC_block = np.column_stack([
            normalize(nac_irrep_proj[j][Γ]["proj_vec"]) for j in chosen
        ])

        # Align subspaces
        ENAC_rot, Sigma = align_subspace(E0_block, ENAC_block)

        # Store results
        for k, to_idx in enumerate(group):
            #e0 = E0_block[:, k]
            #eN = ENAC_rot[:, k]
            #overlap = np.abs(np.vdot(e0, eN))**2
            #results[to_idx] = {
            #    "nac_index": chosen[k],
            #    "aligned_eigenvector": eN,
            #    "overlap": overlap,
            #    "irrep": Γ,
            #    "singular_value": Sigma[k]
            #}
            results[to_idx] = chosen[k]

    return results


def getLOFreqs(path, eigvecs, eigvals, qdir, ordering, degenerates, labels):
    # get the LO modes corresponding to the direction to be analyzed
    #<phonopy --readfc --sym-fc --writedm --qpoints="0 0 0" --nac --q-direction="0 0 1">

    #print(eigvals)

    eigvals_tmp, eigvecs_tmp, norms_pt, qpoint_pt, basis, nat, elements, cPos, masses = parsePhonopy(path, qdir)

    # phonopy always provides all modes
    if ordering == "ascending":
        eigvecs_pt = dict(zip([j for j in range(1,3*nat+1)], eigvecs_tmp))
        eigvals_pt = dict(zip([j for j in range(1,3*nat+1)], eigvals_tmp))
    else:
        eigvecs_pt = dict(zip([3*nat+1-j for j in range(1,3*nat+1)], eigvecs_tmp))
        eigvals_pt = dict(zip([3*nat+1-j for j in range(1,3*nat+1)], eigvals_tmp))

    # match the TO to the LO modes

    LoToDict = LOTOassign(eigvecs, eigvecs_pt, degenerates, labels)

    print(LoToDict)
    
    # reorder the frequencies
    eigvalsLO = {}
    for mode in range(1,3*nat+1):
        print(str(eigvals[mode])+" -> "+ str(eigvals_pt[LoToDict[mode]]))
        eigvalsLO[mode] = eigvals_pt[LoToDict[mode]]
    #

    return eigvals_pt
#

def getChi2(path):
    # from yambo o.xx, test case
    # unit cm/V, gaussian
    xx = np.genfromtxt(path+"oxx", dtype=float)
    xy = np.genfromtxt(path+"oxy", dtype=float)
    xz = np.genfromtxt(path+"oxz", dtype=float)
    yy = np.genfromtxt(path+"oyy", dtype=float)
    yz = np.genfromtxt(path+"oyz", dtype=float)
    zz = np.genfromtxt(path+"ozz", dtype=float)
    # return in m/V, SI
    return 4*np.pi/(3*10e4)*1e-2*[xx, yy, zz, xy, yz, xz]
#

def getLOCorrection(path, born, eps_inf, qdir, vol, w, nat, eigvecs):
    # using Fröhlich formula

    eps_inf_q = np.dot( qdir, np.dot(eps_inf, qdir) )
    corr = np.empty(7, dtype=complex)
    for mode in range(len(eigvecs)):
        tot = []
        for i in range(3):
            for j in range(3):
                tmp1 = 0
                tmp2 = 0
                for atom in range(nat):
                    for x in range(3):
                        tmp1 += born[atom][i][x] * eigvecs[mode][atom][x]
                        tmp2 += born[atom][j][x] * eigvecs[mode][atom][x]
                        #
                    #
                #
                tot.append((4*np.pi/eps_inf_q)**2*tmp1*tmp2)
            #
        #
        tmp3 = [tot[0], tot[4], tot[-1], tot[1], tot[5], tot[2]]
        perp, back = placzeckInvs(tmp3, 1)
        #             xx      yy      zz       xy      yz      xz    perp   back
        corr.append(tot[0], tot[4], tot[-1], tot[1], tot[5], tot[2], perp, back)
    #

    return corr
#                



    """
    chi2 = np.empty((3,3,3), dtype=complex)
    
    chi2[0,0,0] = np.interp([w], [x[0] for x in chi2_tmp[0]], [complex(x[2], x[1]) for x in chi2_tmp[0]])
    chi2[0,0,1] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[2], x[1]) for x in chi2_tmp[3]])
    chi2[0,0,2] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[2], x[1]) for x in chi2_tmp[5]])
    chi2[1,0,0] = np.interp([w], [x[0] for x in chi2_tmp[0]], [complex(x[4], x[3]) for x in chi2_tmp[0]])
    chi2[1,0,1] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[4], x[3]) for x in chi2_tmp[3]])
    chi2[1,0,2] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[4], x[3]) for x in chi2_tmp[5]])
    chi2[2,0,0] = np.interp([w], [x[0] for x in chi2_tmp[0]], [complex(x[6], x[5]) for x in chi2_tmp[0]])
    chi2[2,0,1] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[6], x[5]) for x in chi2_tmp[3]])
    chi2[2,0,2] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[6], x[5]) for x in chi2_tmp[5]])

    chi2[0,1,0] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[2], x[1]) for x in chi2_tmp[3]])
    chi2[0,1,1] = np.interp([w], [x[0] for x in chi2_tmp[1]], [complex(x[2], x[1]) for x in chi2_tmp[1]])
    chi2[0,1,2] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[2], x[1]) for x in chi2_tmp[4]])
    chi2[1,1,0] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[4], x[3]) for x in chi2_tmp[3]])
    chi2[1,1,1] = np.interp([w], [x[0] for x in chi2_tmp[1]], [complex(x[4], x[3]) for x in chi2_tmp[1]])
    chi2[1,1,2] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[4], x[3]) for x in chi2_tmp[4]])
    chi2[2,1,0] = np.interp([w], [x[0] for x in chi2_tmp[3]], [complex(x[6], x[5]) for x in chi2_tmp[3]])
    chi2[2,1,1] = np.interp([w], [x[0] for x in chi2_tmp[1]], [complex(x[6], x[5]) for x in chi2_tmp[1]])
    chi2[2,1,2] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[6], x[5]) for x in chi2_tmp[4]])

    chi2[0,2,0] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[2], x[1]) for x in chi2_tmp[5]])
    chi2[0,2,1] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[2], x[1]) for x in chi2_tmp[4]])
    chi2[0,2,2] = np.interp([w], [x[0] for x in chi2_tmp[2]], [complex(x[2], x[1]) for x in chi2_tmp[2]])
    chi2[1,2,0] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[4], x[3]) for x in chi2_tmp[5]])
    chi2[1,2,1] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[4], x[3]) for x in chi2_tmp[4]])
    chi2[1,2,2] = np.interp([w], [x[0] for x in chi2_tmp[2]], [complex(x[4], x[3]) for x in chi2_tmp[2]])
    chi2[2,2,0] = np.interp([w], [x[0] for x in chi2_tmp[5]], [complex(x[6], x[5]) for x in chi2_tmp[5]])
    chi2[2,2,1] = np.interp([w], [x[0] for x in chi2_tmp[4]], [complex(x[6], x[5]) for x in chi2_tmp[4]])
    chi2[2,2,2] = np.interp([w], [x[0] for x in chi2_tmp[2]], [complex(x[6], x[5]) for x in chi2_tmp[2]])

    #
    # formats
    # chi2[i][j][l]
    # born[atom][l][k]
    # qdir[l]
    # eps_inf[l][k]
    # formula from https://www.nature.com/articles/s41524-024-01236-3

    corr = np.empty(7, dtype=complex)
    dirdict = {0: (0,0), 1: (1,1), 2: (2,2), 3: (0,1), 4: (1,2), 5: (0,2)}
    for dir in range(6):
        tmpEps = 0
        tmpZ = 0
        tmpChi2 = 0
        for j in range(3):
            m, n = dirdict[dir]
            tmpChi2 += chi2[m, n, j]
            for k in range(3):
                tmpEps += qdir[j]*eps_inf[j,k]*qdir[k]
                for atom in range(nat):
                    tmpZ += qdir[j]*born[atom,j,k]
                #
            #
        #
        corr[dir] += 8*np.pi / vol * ( tmpZ * e_charge) / tmpEps * tmpChi2
    #
        
    # units are now 10e-30 C/Vm^2
    """
    
    return corr
#