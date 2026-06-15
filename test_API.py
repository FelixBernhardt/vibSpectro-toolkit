from RamanPy_API import Phonon
from IO import writeData, writeRaman, writeConstantRaman, writeRamanSpectrum, writeIRSpectrum, writeReflectanceSpectrum
import sys

# files created by displace in ascending or descending order is important!! Here, all use the VASP default
# reordering works like this
#test.modelist=[30-mode for mode in range(30)][3:]
#test.eigenvecs=[test.eigenvecs[29-mode] for mode in range(30)]
#test.eigenfreqs=[test.eigenfreqs[29-mode] for mode in range(30)]

path = "/home/felix/Forschung/sqs_00"
test = Phonon(file="OUTCAR", path=path, born=True, qdir=(0,1,0), nosym=False, modelist=range(1,31), smearing=5, LOcorr=True)


#print(test.pointgroup)
#test.print_decomposition()
#print(test.eigenfreqs)
#print(test.ordering)
#print(test.labels)
#print(test.degenerates)
#print(test.acoustics)
#print(test.silent)

#test.IR()
#test.reflectance()
#test.write_System()
#test.write_IR()
#test.write_Reflectance()
#test.load_IR()
#test.load_Reflectance()
#test.plot_IR()
#test.plot_reflectance()

#test.pointgroup = "3m"
#from Symmetries import analyzeRamanTensors
#test.ramantensors = analyzeRamanTensors(test.pointgroup)
#test.print_ramanselection()
#test.calc_raman_tensors()
#test.write_tensors()
test.load_raman_tensors()
test.calc_raman_spectrum()
#test.plot_raman()
#test.write_spectrum()
#test.write_tensors()
#test.load_ramantensors_data(5)


#print(test.ramanspectrum_data[2][0:10])
#test.load_spectrum()
#print(test.ramanspectrum_data[2][0:10])
#test.plot_Raman()

#test.pointgroup = "6"
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