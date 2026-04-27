from RamanPy_API import Phonon

positions=[[ 0.000000,  0.000000,  0.000000],
           [ 0.000000,  0.000000,  1.561000],
           [ 0.000000,  1.561000,  0.000000],
           [ 0.000000,  0.000000, -1.561000],
           [ 0.000000, -1.561000,  0.000000],
           [ 1.561000,  0.000000,  0.000000],
           [-1.561000,  0.000000,  0.000000]], 
symbols=['S', 'F', 'F', 'F', 'F', 'F', 'F']

path = "/home/felix/Forschung/molecule/"
test = Phonon(code_in="VASP", path=path, born="VASP", qdir=(1,0,0), nosym=True, molecule=True)
test.cartesian = positions
test.elements = symbols
test.set_symmetries()

#print(test.ordering)
print(test.labels)
print(test.acoustics)
print(test.silent)
print(test.degenerates)
#print(test.modelist)
#test.print_ramantensors()
#test.print_dielectrictensor()
#test.print_irselection()
#test.print_ramanselection()
#test.displace()
#test.IR()
#test.plotIR(lualatex=False)

test.tensors()
test.spectrum()
test.plotRaman(lualatex=False)