#!/usr/bin/env python

from src.API import Phonon

path = "/home/felix/Downloads/RamanPy/examples/LiNbO3"
file = "OUTCAR"
LiNbO3 = Phonon(name="LiNbO3", file=file, path=path, born=True, symprec=1.e-3, degeneracy_tolerance=5.e-2)

LiNbO3.calc_raman_displace_overtones()