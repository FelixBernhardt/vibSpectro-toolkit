from RamanPy_API import *

path = "/home/felixbernhardt/Forschung/sqs_00_VASP/"
#path = "/home/felixbernhardt/Forschung/sqs_00_allVASP/"
#path = "/home/felixbernhardt/Forschung/Yassine/"
test = Phonon(path=path, born="VASP")

print(test.labels)
print(test.eigenfreqs)
print(test.acoustics)
print(test.silent)
print(test.degenerates)
print(test.modelist)
#test.print_ramantensors()
#test.print_dielectrictensor()
#test.print_irselection()
#test.print_ramanselection()
#test.displace()

#test.IR(plotFlag=True)
test.tensors()