import numpy as np
import re
import sys

def calcDegenerates(pointgroup):
    # 3m
    R1 = np.array([["c", 0, 0], [0, "-c", "d"], [0, "d", 0]], dtype=str)
    R2 = np.array([[0, "c", "d"], ["c", 0, 0], ["d", 0, 0]], dtype=str)

    # -43m
    RE1 = np.array([["b", 0, 0], [0, "b", 0], [0, 0, "-2b"]], dtype = "str")
    RE2 = np.array([["-\u221A3b", 0, 0], [0, "\u221A3b", 0], [0, 0, 0]], dtype = "str")
    RT1 = np.array([[0, 0, 0], [0, 0, "d"], [0, "d", 0]], dtype = "str")
    RT2 = np.array([[0, 0, "d"], [0, 0, 0], ["d", 0, 0]], dtype = "str")
    RT3 = np.array([[0, "d", 0], ["d", 0, 0], [0, 0, 0]], dtype = "str")

    print(R1)
    print(R2)
    #print(RE1)
    #print(RE2)
    #print(RT1)
    #print(RT2)
    #print(RT3)

    #Rn = [RE1, RE2]
    #Rn = [RT1, RT2, RT3]
    Rn = [R1, R2]
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

    # orthonormalization, E holds the basis matrizes that the Ramantensor decomposes into
    M = np.column_stack([R.reshape(-1) for R in Rb])
    U, S, Vt = np.linalg.svd( M, full_matrices=False)
    E = []
    for i in range(len(U[0])):
        E.append(U[:, i].reshape(3,3))
        print(U[:, i].reshape(3,3))
    #


    # get the calculated, degenerate mode
    #data = np.genfromtxt("/Volumes/MacintoshHD-Daten/Users/felixbernhardt/LNT/LNT_IR/Raman/sqs_00/alpha_5.dat", dtype=complex)
    data = np.genfromtxt("alpha_5.dat", dtype=complex)

    for i in [0]:
    #for i in range(len(data)):
        if data[i,7] > 1.e-1:
            Atest = np.zeros((3,3))
            Atest[0,0] = np.real(data[i,1])
            Atest[1,1] = np.real(data[i,2])
            Atest[2,2] = np.real(data[i,3])
            Atest[0,1] = np.real(data[i,4])
            Atest[1,2] = np.real(data[i,5])
            Atest[0,2] = np.real(data[i,6])
            Atest[1,0] = np.real(Atest[0,1])
            Atest[2,1] = np.real(Atest[1,2])
            Atest[2,0] = np.real(Atest[0,2])

            #Atest = np.array([[1-np.sqrt(3),0,0],[0,1+np.sqrt(3),0],[0,0,-2]])
            #Atest = np.array([[0,1,2],[1,0,3],[2,3,0]])
            print(Atest)

            x = np.array( np.linalg.lstsq( np.column_stack([tmp.reshape(-1) for tmp in E]), Atest.reshape(-1), rcond=None)[0] )
            print(x)

            # create degenerate ramantensors
            u = np.linalg.svd(x.reshape(-1, 1), full_matrices=True)[0]
            prefactor = [x[i]/u[0,i] for i in range(len(x))]

            degenerates = []
            for i in range(len(Rn)):
                R_degen = np.zeros((3,3))
                for j in range(len(x)):
                    R_degen = np.add(R_degen, prefactor[j]*u[i,j]*E[j])
                #
                print(R_degen)
            #
        #
    return degenerates
    #
#


# test calc
degenerates = calcDegenerates(pointgroup)