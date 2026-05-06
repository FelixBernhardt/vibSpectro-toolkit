from RamanPy_API import Phonon
from RamanLib import analyzeDielectricTensor, analyzeRamanTensors

# files created by displace in ascending or descending order is important!! Here, all use the VASP default
# reordering works like this
#test.modelist=[30-mode for mode in range(30)][3:]
#test.eigenvecs=[test.eigenvecs[29-mode] for mode in range(30)]
#test.eigenfreqs=[test.eigenfreqs[29-mode] for mode in range(30)]

#path = "/Users/felixbernhardt/Desktop/sqs_00"
path = "/home/felix/Forschung/test_molecule"
path = "/home/felix/Forschung/sqs_00"
#path = "/home/felixbernhardt/Forschung/sqs_00_allVASP/"
#path = "/home/felixbernhardt/Forschung/Yassine/"
test = Phonon(file="phonopy.yaml", path=path, born=False, qdir=(0,1,0), nosym=False, modelist=[i for i in range(1,31)], smearing=20)

#print(test.ordering)
#print(test.labels)
#print(test.acoustics)
#print(test.silent)
#print(test.degenerates)
#print(test.modelist)
test.pointgroup = "432"
test.ramantensors = analyzeRamanTensors(test.pointgroup, varprint=False)
test.dielectrictensor = analyzeDielectricTensor(test.pointgroup, varprint=False)
test.print_ramantensors()
test.print_dielectrictensor()
test.print_irselection()
test.print_ramanselection()
#test.displace()
#test.IR()
#test.plotIR(lualatex=False)
#test.reflectance()
#test.plotReflectance()
#test.tensors()
#test.spectrum()
#test.plotRaman(lualatex=False)