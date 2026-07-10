from RamanPy_API import Phonon
from Datastruct import writeData, writeRaman, writeConstantRaman, writeRamanSpectrum, writeIRSpectrum, writeReflectanceSpectrum
import sys

path = "/Users/felixbernhardt/Desktop/DHTAP"
test = Phonon(file="OUTCAR", path=path, born=True, qdir=(1,1,0), nosym=True, smearing=5, modelist=range(1,97))

test.IR()
test.plot_IR()
test.reflectance()
test.plot_reflectance()
writeIRSpectrum(test)
writeReflectanceSpectrum(test)
sys.exit(0)
