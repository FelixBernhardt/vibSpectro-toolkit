import sys, os
pwd = os.getcwd()
sys.path.append(pwd+"/..") # set to your installation path
from src.RamanPy_API import Phonon

path = pwd+"/LiNbO3"
file = "OUTCAR"
LiNbO3 = Phonon(name="LiNbO3", file=file, path=path, born=True)

LiNbO3.set_symmetries(LiNbO3.modelist)
print(LiNbO3.labels)
LiNbO3.print_decomposition()
LiNbO3.print_ramantensors()
LiNbO3.print_ramanselection()
print(LiNbO3.acoustics)