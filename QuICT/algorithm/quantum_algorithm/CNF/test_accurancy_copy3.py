import os
import time
import numpy as np
import math
import random
from QuICT.algorithm.quantum_algorithm.CNF.cnf import *#CNFSATOracle
from QuICT.qcda.optimization import *
# from QuICT.simulation.unitary_simulator import UnitarySimulator
from QuICT.simulation.cpu_simulator import CircuitSimulator


def read_CNF(cnf_file):
        # file analysis
        #输入一个文件，输出CNF格式的list
        variable_number = 0
        clause_number = 0
        CNF_data = []
        f = open(cnf_file, 'r') 
        for line in f.readlines():
            new = line.strip().split()
            int_new=[]
            if new[0] == 'p':
                variable_number = int(new[2])
                clause_number = int(new[3])
            else:
                for i in range(len(new)-1): #注意这里减1 
                    int_new.append(int(new[i]))
            CNF_data.append(int_new)  #给各个Clause 编号0,1 ...m-1#
        #print(CNF_data)
        f.close()
        return variable_number, clause_number, CNF_data

def test():
    # x0 x1，x2, x_{n variable_number -1}
    filename_test =  "QuICT/algorithm/quantum_algorithm/CNF/test_data/6_9_0"
    AuxQubitNumber = 4
    variable_number , clause_number , CNF_data = read_CNF(filename_test)


    #真值表初值变化
    b=[]
    # print(variable_number)
    cnf = CNFSATOracle()
    cnf.run(filename_test, AuxQubitNumber)
    cgate = cnf.circuit()
    circuit_temp = Circuit(variable_number + 2 + AuxQubitNumber)
    d=random.sample(list(range(2**variable_number)), 10)
    

    for a in d:
        randomnum = a
        x=[]
        for ii in range(variable_number):
            oneorzero = math.floor(randomnum % 2)
            randomnum = math.floor(randomnum/2) 
            if (oneorzero == 0):
                x.append(0)
            else:
                x.append(1)
                X | circuit_temp(ii)
        circuit_temp.extend(cgate)
        Measure | circuit_temp
        print(circuit_temp.width())
        print(circuit_temp.size())

        stime = time.time()
        sim = CircuitSimulator()
        amplitude = sim.run(circuit_temp)
        ltime = time.time()
        print(f"cnf run speed : {ltime - stime}")

        #print(circuit_temp.qubits[variable_number].measured)
        #print(int(circuit_temp.qubits[variable_number]))
            

        cnf_result = 1  #经典部分 真假值 
        for i in range(clause_number):
            clause_result = 0
            for j in range(len(CNF_data[i+1])):
                if CNF_data[i+1][j] > 0:
                    clause_result += x[CNF_data[i+1][j]-1] 
                else:
                    if CNF_data[i+1][j] < 0:
                        clause_result += (1 - x[-CNF_data[i+1][j]-1] )
            if clause_result == 0:
                cnf_result = 0
                break
        print(cnf_result, " ", a)

        if cnf_result !=  circuit_temp.qubits[variable_number].measured : #比较一下经典与量子电路的 真假值(是否满足的情况)，是否相同。
            print("!!!!!!!!!")
            print(a)
            print(circuit_temp.qubits[variable_number].measured)
            print(cnf_result)
            b.append(a)
    print(b)
    # f.close()
test()