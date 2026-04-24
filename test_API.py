from RamanPy_API import *

# plots use lualatex !!
# files created by displace in ascending or descending order is important!! Here, all use the VASP default
# reordering works like this
#test.modelist=[30-mode for mode in range(30)][3:]
#test.eigenvecs=[test.eigenvecs[29-mode] for mode in range(30)]
#test.eigenfreqs=[test.eigenfreqs[29-mode] for mode in range(30)]

path = "/Users/felixbernhardt/Desktop/sqs_00"
#path = "/home/felix/Forschung/sqs_00/"
#path = "/home/felixbernhardt/Forschung/sqs_00_allVASP/"
#path = "/home/felixbernhardt/Forschung/Yassine/"
test = Phonon(code_in="VASP", path=path, born="VASP", plot=True, qdir=(1,0,0))

#print(test.ordering)
#print(test.labels)
#print(test.acoustics)
#print(test.silent)
#print(test.degenerates)
#print(test.modelist)
#test.print_ramantensors()
#test.print_dielectrictensor()
#test.print_irselection()
#test.print_ramanselection()
#test.displace()
test.IR()

test.tensors()
test.spectrum()