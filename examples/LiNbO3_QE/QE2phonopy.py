#!/usr/bin/env python

# from https://phonopy.github.io/phonopy/qe.html

import sys
from phonopy.interface.qe import read_pwscf, PH_Q2R

primcell_filename = sys.argv[1] # "scf.in"
q2r_filename = sys.argv[2]      # "matdyn.fc"
cell, _ = read_pwscf(primcell_filename)
q2r = PH_Q2R(q2r_filename)
q2r.run(cell)
q2r.write_force_constants(fc_format="dummy") # just dont put hdf5 for fc_format, also q2r.save() seems deprecated...