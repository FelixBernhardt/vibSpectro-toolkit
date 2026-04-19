import numpy as np
import re
from RamanLib import placzeck

# something is wrong here!

def calcDegenerates(path, modes, labels, ramantensors):
    # get the corresponding ramantensors
    Rn = []
    label = labels[modes[0]-1]
    for i in range(int(len(ramantensors)/2)):
        if ramantensors[2*i] == label:
            Rn.append(ramantensors[2*i+1])
        #
    #

    #print(Rn)
    Rb = []
    # decompose general tensors into their coefficients
    for letter in ["a", "b", "c", "d", "e", "f"]:
        Rnc = []
        for R in Rn:
            foundOne = False
            tmp_R = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    tmp = re.findall(letter, R[i,j])
                    if tmp != []:
                        foundOne = True
                        tmp = re.split(letter, R[i,j])
                    
                        if tmp[0] == "":
                            tmp_R[i,j] = 1
                        elif tmp[0] == "-":
                            tmp_R[i,j] = -1
                        elif len(tmp[0]) > 1:
                            if tmp[0][0] == "\u221A":
                                tmp_R[i,j] = np.sqrt(float(tmp[0][1]))
                            elif tmp[0][1] == "\u221A" and tmp[0][0] == "-":
                                tmp_R[i,j] = -np.sqrt(float(tmp[0][2]))
                            else:
                                tmp_R[i,j] = -float(tmp[0][-1])
                            #
                        else:
                            tmp_R[i,j] = float(tmp[0][-1])
                        #
                    #
                #
            #
        
            if foundOne == True:
                Rnc.append(tmp_R)
                Rb.append(np.array(tmp_R))
            #
        #
    #

    # orthonormalization, E holds the basis matrizes that the calculated Ramantensor decomposes into
    M = np.column_stack([R.reshape(-1) for R in Rb])
    U, S, Vt = np.linalg.svd( M, full_matrices=False)
    E = []
    for i in range(len(U[0])):
        E.append(U[:, i].reshape(3,3))
        #print(E[-1])
    #

    # get the calculated, degenerate raman tensor
    data = np.genfromtxt(path+"Ramantensors/alpha_"+str(modes[0])+".dat", dtype=complex)
    with open(path+"Ramantensors/alpha_"+str(modes[0])+".dat") as f:
        f.readline()
        eigval = f.readline().split()[-1]
    #

    w = []
    I = []
    #for i in [0]:
    for i in range(len(data)):
        w.append(np.real(data[i,0]))
        Atest = np.zeros((3,3), dtype=complex)
        Atest[0,0] = data[i,1]
        Atest[1,1] = data[i,2]
        Atest[2,2] = data[i,3]
        Atest[0,1] = data[i,4]
        Atest[1,2] = data[i,5]
        Atest[0,2] = data[i,6]
        Atest[1,0] = Atest[0,1]
        Atest[2,1] = Atest[1,2]
        Atest[2,0] = Atest[0,2]

        #print(Atest)

        x = np.array( np.linalg.lstsq( np.column_stack([tmp.reshape(-1) for tmp in E]), Atest.reshape(-1), rcond=None)[0] )
        #print(x)

        # create degenerate ramantensors
        u = np.linalg.svd(x.reshape(-1, 1), full_matrices=True)[0]
        prefactor = [x[i]/u[0,i] if u[0,i] != 0 else 0 for i in range(len(x))]

        degenerates = np.zeros((len(Rn),6,1), dtype=complex)
        for j in range(len(Rn)):
            R_degen = np.zeros((3,3))
            for k in range(len(x)):
                R_degen = np.add(R_degen, prefactor[k]*u[j,k]*E[k])
            #
            degenerates[j] = [[R_degen[0,0]], [R_degen[1,1]], [R_degen[2,2]], [R_degen[0,1]], [R_degen[1,2]], [R_degen[0,2]]]
        #
        I.append(degenerates)
    #
    I = np.array(I, dtype=complex)

    # write to file
    for j in range(0,len(modes)):
        outfile = path+"Ramantensors/alpha_"+str(modes[j])+".dat"
        f = open(outfile, "w")
        f.write("# Raman tensor in 10^(-30) Cm^2/V\n")
        f.write("# mode: " +str(modes[j])+"   phonon freq: "+str(eigval)+"\n")
        f.write("# omega(eV)    xx        yy        zz        xy        yz        xz      avg\n")

        for i in range(len(w)):
            avg = placzeck( I[i][j], 1)
            f.write("{:5.5f} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3f}\n"\
                .format(w[i], I[i][j][0][0], I[i][j][1][0], I[i][j][2][0], I[i][j][3][0], I[i][j][4][0], I[i][j][5][0], avg))
            #
        #
        f.close()
    #
#