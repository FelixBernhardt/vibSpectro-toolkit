#!/usr/bin/env python

from src.API import Phonon
from src.Symmetries import RamanTensorComponents2ndorder

overtone_dict = {str(["A2", "A2"]): "A1"}

path = "/home/felix/Downloads/RamanPy/examples/LiNbO3"
file = "OUTCAR"
LiNbO3 = Phonon(name="LiNbO3", file=file, path=path, born=True, symprec=1.e-3, degeneracy_tolerance=5.e-2)

LiNbO3.calc_raman_displace_overtones()
j = 27
print([LiNbO3.labels[LiNbO3.modelistovertones[j][i]] for i in range(2)])

print("modes:", LiNbO3.modelistovertones[j])
print("labels:", RamanTensorComponents2ndorder[LiNbO3.pointgroup][str([LiNbO3.labels[LiNbO3.modelistovertones[j][i]] for i in range(2)])])
print("ramantensors:", LiNbO3.ramantensors )

LiNbO3.calc_raman_tensors_overtones()