import os
import time
from math import pi, sqrt
import numpy as np
import cupy as cp
import math
import random
from QuICT.algorithm.quantum_algorithm.CNF.cnf import *#CNFSATOracle
from QuICT.qcda.optimization import *
# from QuICT.simulation.unitary_simulator import UnitarySimulator
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_algorithm.grover import (
    Grover
)
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux

def read_CNF(cnf_file):
        # file analysis
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
        #print(CNF_data, "classical")
        f.close()
        return variable_number, clause_number, CNF_data

def check_solution(variable_data, variable_number, clause_number, CNF_data):
    cnf_result = 1  #经典部分 真假值 
    for i in range(clause_number):
        clause_result = 0
        if CNF_data[i+1] == []:
            clause_result = 1
        else:
            for j in range(len(CNF_data[i+1])):
                if CNF_data[i+1][j] > 0:
                    clause_result = clause_result + variable_data[CNF_data[i+1][j]-1]
                else:
                    if CNF_data[i+1][j] < 0:
                        clause_result = clause_result  + ( 1 - variable_data[-CNF_data[i+1][j]-1] )
            if clause_result == 0:
                cnf_result = 0
                break
    if cnf_result == 1:
        return True
    else:
        return False

def find_solution_count(filename_test):
    solutions = []
    variable_number , clause_number , CNF_data = read_CNF(filename_test)
    for i in range(1<<variable_number):
        variable_data = bin(i)[2:].rjust(variable_number,'0')[::-1]
        variable_data = [int(x) for x in variable_data]
        if check_solution(variable_data, variable_number, clause_number, CNF_data):
            solutions.append(variable_data)
    # print(solutions)
    return len(solutions)

def test(filename_test, variable_number, clause_number, CNF_data, runs):
    AuxQubitNumber = 5
    cnf = CNFSATOracle()
    cnf.run(filename_test, AuxQubitNumber, 1)
    n_hit = 0
    oracle = cnf.circuit()
    grover = Grover(ConstantStateVectorSimulator())
        
    circ = grover.circuit(variable_number, AuxQubitNumber + 1, oracle, n_solution, measure=False, is_bit_flip=True)
    grover.simulator.run(circ)
    result_samples = grover.simulator.sample(runs)
    result_var_samples = np.array(result_samples).reshape((1<<variable_number,1<<(AuxQubitNumber + 1))).sum(axis=1)
    for result in range(1<<variable_number):
        result_str = bin(result)[2:].rjust(variable_number,'0')
        if check_solution([int(x) for x in result_str], variable_number, clause_number, CNF_data):
            n_hit += result_var_samples[result]
        #     print("True",end='\r')
        # else:
        #     print("False",end='\r')
    return n_hit

file_dir = '/home/wangrui/quict/QuICT/algorithm/quantum_algorithm/CNF/cnf_test/'
filename_test_list = os.listdir(file_dir)
filename_test_list.sort()
i = 0
l = len(filename_test_list)
n_all = 50000
n_all_all = n_all*l
n_hit_all = 0
result = []
for filename_test in filename_test_list:
    i+=1
    file_path = file_dir+filename_test
    n_solution = find_solution_count(file_path)
    variable_number , clause_number , CNF_data = read_CNF(file_path)
    print(f"[{i:3}/{l:3}]{n_solution:4} solution in {1<<variable_number:4} possibility")
    # if n_solution==0:
    #     n_hit = None
    #     print(f"{filename_test:10} with zero solution")
    # else:
    #     n_hit = test(file_path, variable_number, clause_number, CNF_data, n_all)
    #     print(f"{filename_test:10} success rate: {n_hit/n_all:5.3f}[{n_hit:3}/{n_all:3}]")

#     # print(n_solution)
#     result.append([
#         filename_test,
#         n_all,
#         n_hit,
        
#     ])
# import pandas as pd
# pd.DataFrame(result).to_csv("cnf_test_result")

# print(math.asin(sqrt(2/65536)))
# print(round(pi/4/0.005524299-0.5))