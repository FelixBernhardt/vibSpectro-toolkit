from RamanPy_API import Phonon
from Datastruct import data, writeData, writeRaman, writeConstantRaman, writeSpectrum
from RamanLib import analyzeDielectricTensor, analyzeRamanTensors

# files created by displace in ascending or descending order is important!! Here, all use the VASP default
# reordering works like this
#test.modelist=[30-mode for mode in range(30)][3:]
#test.eigenvecs=[test.eigenvecs[29-mode] for mode in range(30)]
#test.eigenfreqs=[test.eigenfreqs[29-mode] for mode in range(30)]

#path = "/Users/felixbernhardt/Desktop/sqs_00"
#path = "/home/felix/Forschung/test_molecule"
path = "/home/felix/Forschung/sqs_00"
#path = "/home/felixbernhardt/Forschung/sqs_00_allVASP/"
#path = "/home/felixbernhardt/Forschung/Yassine/"
#test = Phonon(file="phonopy.yaml", path=path, born=False, qdir=(0,1,0), nosym=False, modelist=[7,8], smearing=5)
test = Phonon(file="phonopy.yaml", path=path, born=True, qdir=(0,1,0), nosym=False, modelist=range(1,31), smearing=5)
#test = Phonon(file="OUTCAR", path=path, born=False, qdir=(1,0,0), nosym=True, modelist=[2,3,4,5,6,8,9,11,12,13,14,15,16,17,19,20,21,22,23,24,26,27], smearing=5)

print(test.pointgroup)
test.print_decomposition()
yamlData = data(test)
writeData(test)
test.tensors()
writeRaman(test)
test.spectrum()
writeConstantRaman(test)
writeSpectrum(test)

#print(test.ordering)
#print(test.labels)
#print(test.acoustics)
#print(test.rotations)
#print(test.silent)
#print(test.degenerates)
#print(test.modelist)
#test.print_decomposition()
#test.pointgroup = "6"
#test.ramantensors = analyzeRamanTensors(test.pointgroup, varprint=False)
#test.dielectrictensor = analyzeDielectricTensor(test.pointgroup, varprint=False)
#test.print_ramantensors()
#test.print_dielectrictensor()
#test.print_irselection()
#test.print_ramanselection()
#test.displace()
#test.IR()
#test.plotIR(lualatex=False)
#test.reflectance()
#test.write_Reflectance()
#test.plotReflectance()
#test.tensors()
#test.write_tensors()
#print(test.ramantensors_data)
#test.spectrum()
#print(test.ramanspectrum_data[0])
#test.write_spectrum()
#test.plot_Raman(lualatex=False)