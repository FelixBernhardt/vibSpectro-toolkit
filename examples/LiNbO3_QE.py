#!/usr/bin/env python

from src.RamanPy_API import Phonon
import numpy as np

path = "/home/felix/Downloads/RamanPy/examples/LiNbO3_QE"
file = "scf.out"
#file = "phonopy.yaml"

# FORCE_SETS can be created by running q2r.x
# for gamma only calculations, just copy matdyn to matdyn1, remove the dielectric tensor and effective charges from matdyn1
# and provide a file matdyn0 with the q-point "grid" (see the matdyn0 file in the folder)
# then you can run q2r.x -i q2r.in
# afterwards, run QE2phonopy scf.in matdyn.fc
# create BORN file phonopy-qe-born scf.in ph.out > BORN
# create phonopy.yaml and qpoints.yaml by phonopy --readfc --qe -c="scf.in" --dim="1 1 1" --writedm --qpoints="0 0 0" --nac
# if the symmetry and mode labels are not correctly recognized, fix this by adjusting the symprec and degeneracy_tolerance parameters
# either when running phonopy (recommended) or when initializing the Phonon object 

LiNbO3 = Phonon(file=file, path=path, code_out="QE", born=True, qdir=(1,0,0), modelist=range(1,31), nosym=False, symprec=1.e-3, degeneracy_tolerance=5.e-4)

# calculate and plot the IR spectrum
# symmetries are not considered here, as the calculations are very fast anyway
LiNbO3.calc_ir()
LiNbO3.calc_reflectance()
LiNbO3.write_system()
LiNbO3.write_ir()
LiNbO3.write_reflectance()
LiNbO3.plot_ir()
LiNbO3.plot_reflectance()

# calculate and plot the Raman spectrum in x(..)-x configuration, no LO-TO splitting needs to be considered
# check if the correct symmetries are recognized, and that all modes can be assigned a label (acoustic modes do not need a label)
# also double check the selection rules for LO-TO splitting
LiNbO3.print_pointgroup()
LiNbO3.print_decomposition()
LiNbO3.print_ramanselection()
LiNbO3.calc_raman_displace()
#LiNbO3.calc_raman_tensors()
#LiNbO3.write_raman_tensors()
#LiNbO3.calc_raman_spectrum()
#LiNbO3.plot_raman()
#LiNbO3.write_raman_spectrum()