import os
import time
import numpy as np
import math
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
    filename_test =  "QuICT/algorithm/quantum_algorithm/CNF/2.cnf"
    variable_number , clause_number , CNF_data = read_CNF(filename_test)


    #真值表初值变化
    for a in range(2 ** variable_number):  #改
        cnf = CNFSATOracle()
        cnf.run(filename_test)
        cgate = cnf.circuit()
        circuit_temp=Circuit(variable_number + 2 + 4)
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

        # circuit_temp.draw(filename='test_0.jpg')
        # sim = CircuitSimulator()
        # amplitude = sim.run(circuit_temp)
        print(int(circuit_temp.qubits[variable_number])) #量子结果
        result_lz = int(circuit_temp.qubits[variable_number])

        cnf_result = 1
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
        print(cnf_result) #经典结果
        assert result_lz == cnf_result
test()