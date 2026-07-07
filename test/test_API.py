from RamanPy_API import Phonon
from IO import writeData, writeRaman, writeConstantRaman, writeRamanSpectrum, writeIRSpectrum, writeReflectanceSpectrum

# files created by displace in ascending or descending order is important!! Here, all use the VASP default
# reordering works like this
#test.modelist=[30-mode for mode in range(30)][3:]
#test.eigenvecs=[test.eigenvecs[29-mode] for mode in range(30)]
#test.eigenfreqs=[test.eigenfreqs[29-mode] for mode in range(30)]

#path = "/Users/felixbernhardt/Desktop/sqs_00"
path = "/home/felix/Forschung/sqs_00"
test = Phonon(file="OUTCAR", path=path, born=True, qdir=(0,0,1), nosym=False, modelist=range(1,31), smearing=5, LOcorr=True)


#print(test.pointgroup)
#test.print_decomposition()
#print(test.eigenfreqs)
#print(test.ordering)
#print(test.labels)
print(test.degenerates)
#print(test.acoustics)
#print(test.silent)

"""
test.calc_ir()
test.calc_reflectance()
test.write_system()
test.write_ir()
test.write_reflectance()
test.load_ir()
test.load_reflectance()
test.plot_ir()
test.plot_reflectance()

#test.pointgroup = "3m"
#from Symmetries import analyzeRamanTensors
#test.ramantensors = analyzeRamanTensors(test.pointgroup)
test.print_ramanselection()
"""
test.calc_raman_tensors()
test.write_raman_tensors()
#test.load_raman_tensors()
test.calc_raman_spectrum()
test.plot_raman()
test.write_raman_spectrum()