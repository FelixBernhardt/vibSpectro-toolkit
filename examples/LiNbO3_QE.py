#!/usr/bin/env python

from src.RamanPy_API import Phonon

path = "LiNbO3_QE"
file = "scf.out"
LiNbO3 = Phonon(file=file, path=path, code_out="QE", born=True, qdir=(1,0,0), modelist=range(4,31), smearing=5, LOcorr=False)

# Symmetry information are not provided as FORCE_SETS has not been created. Refer to the phonopy wiki on how to create FORCE_SETS from QE
print(LiNbO3.pointgroup)
LiNbO3.print_decomposition()

# calculate and plot the IR spectrum
LiNbO3.calc_ir()
LiNbO3.calc_reflectance()
LiNbO3.write_system()
LiNbO3.write_ir()
LiNbO3.write_reflectance()
LiNbO3.plot_ir()
LiNbO3.plot_reflectance()

# calculate and plot the Raman spectrum in x(..)-x configuration, since LO-TO splitting requires phonopy for now
LiNbO3.print_ramanselection()
LiNbO3.calc_raman_tensors()
LiNbO3.write_raman_tensors()
LiNbO3.calc_raman_spectrum()
LiNbO3.plot_raman()
LiNbO3.write_raman_spectrum()