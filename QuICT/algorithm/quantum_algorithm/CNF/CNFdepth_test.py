import os
import time
import numpy as np
import cupy as cp
import math
import random
from QuICT.algorithm.quantum_algorithm.CNF.cnfdepth import *#CNFSATOracle
from QuICT.qcda.optimization import *
# from QuICT.simulation.unitary_simulator import UnitarySimulator
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


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
                for x in new:
                    if (x != '0') and (int(x) not in int_new):
                        int_new.append(int(x))
                        if (- int(x)) in int_new:
                            int_new = []
                            break
            CNF_data.append(int_new)  #给各个Clause 编号0,1 ...m-1#
        f.close()
        # print(CNF_data)
        return variable_number, clause_number, CNF_data

def test():
    i_list = [6, 10, 14, 18, 22, 26, 30]
    j_list = [6]
    m_list = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    
    # x0 x1，x2, x_{n variable_number -1}
    for i in i_list:
        for j in j_list:
            filename = f"{j}_{i}_5"
            filename_test =  "QuICT/algorithm/quantum_algorithm/CNF/test_data/宽度_变量数_子句数/" + filename
            AuxQubitNumber = 15
            variable_number , clause_number , CNF_data = read_CNF(filename_test)

            #真值表初值变化
            b=[]
            # print(variable_number)
            cnf = CNFSATDEPTHOracle()
            cnf.run(filename_test, AuxQubitNumber)
            cgate = cnf.circuit()
            print(f"{j}_{i}_5")
            print(cgate.size())
            print(cgate.depth())
            
            # circ = Circuit(variable_number + 4)
            
            d=random.sample(list(range(2**variable_number)), 8)
            for a in d+[105, 315, 501, 255, 266, 221, 387, 339, 181, 33]:
                circ = Circuit(variable_number + 1 + AuxQubitNumber)
                x = []
                randomnum = a
                for ii in range(variable_number):
                    oneorzero = math.floor(randomnum % 2)
                    randomnum = math.floor(randomnum/2) 
                    if (oneorzero == 0):
                        x.append(0)
                    else:
                        x.append(1)
                        X | circ(ii)
                cgate | circ
                Measure | circ
                # circuittt.extend(cgate)
                #circ.draw(filename='15depth.jpg')
                # i_sv = cp.zeros(1 << (variable_number + 4), dtype=np.complex64)
                sim = ConstantStateVectorSimulator()
                amplitude = sim.run(circ)

                #print(circuit_temp.qubits[variable_number].measured)
                #print(int(circuit_temp.qubits[variable_number]))

                # x = [0] * (2 ** variable_number)
                # x[a] = 1
                # for i in range(variable_number):
                #     if (1 << i) & a == 1:
                #         x[]
                cnf_result = 1  #经典部分 真假值 
                for i in range(clause_number):
                    clause_result = 0
                    if len(CNF_data[i+1]) == 0:
                        clause_result = 1
                    else:
                        for j in range(len(CNF_data[i+1])):
                            if CNF_data[i+1][j] > 0:
                                clause_result += x[CNF_data[i+1][j]-1] 
                            else:
                                if CNF_data[i+1][j] < 0:
                                    clause_result += (1 - x[-CNF_data[i+1][j]-1] )
                        if clause_result == 0:
                            cnf_result = 0
                            break
                # print(cnf_result, " ", a)

        if cnf_result !=  circ.qubits[variable_number].measured : #比较一下经典与量子电路的 真假值(是否满足的情况)，是否相同。
            print("!!!!!!!!!")
            # print(a)
            # print(circ.qubits[variable_number].measured)
            # print(cnf_result)
            # b.append(a)
    # print(b)
    # f.close()
test()