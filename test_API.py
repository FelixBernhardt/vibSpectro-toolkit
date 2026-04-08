from RamanPy_API import *

#path = "/home/felixbernhardt/Forschung/sqs_00_VASP/"
#path = "/home/felixbernhardt/Forschung/sqs_00_allVASP/"
path = "/home/felixbernhardt/Forschung/Yassine/"
test = Phonon(path=path, code_in="VASP", modelist=range(1,5), nosym=True)

#print(test.labels)
print(test.eigenfreqs)
#print(test.acoustics)
#print(test.silent)
#print(test.degenerates)
#test.print_ramantensors()
#test.print_dielectrictensor()
#test.print_irselection()
#test.print_ramanselection()
test.displace()