from RamanPy_API import *

# plots use lualatex !!

path = "/home/felix/Forschung/sqs_00/"
#path = "/home/felixbernhardt/Forschung/sqs_00_allVASP/"
#path = "/home/felixbernhardt/Forschung/Yassine/"
test = Phonon(code_in="phonopy", path=path, born="VASP", plot=True, qdir=(1,0,0))

print(test.ordering)
print(test.labels)
#print(test.eigenfreqs)
print(test.acoustics)
print(test.silent)
print(test.degenerates)
print(test.modelist)
#test.print_ramantensors()
#test.print_dielectrictensor()
#test.print_irselection()
#test.print_ramanselection()
#test.displace()
#test.IR()
#test.tensors()
test.spectrum()