import os
import time
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
    return len(solutions)

def test():
    filename_test =  "./2"
    AuxQubitNumber = 5
    variable_number , clause_number , CNF_data = read_CNF(filename_test)
    #print(CNF_data)
    #真值表初值变化
    #b=[]
    # print(variable_number)
    n_solution = find_solution_count(filename_test)
    print(f"{n_solution:4} solution in {1<<variable_number:4} possibility")
    cnf = CNFSATOracle()
    cnf.run(filename_test, AuxQubitNumber, 1)
    oracle = cnf.circuit()
    grover = Grover(ConstantStateVectorSimulator())
    
    result = grover.run(variable_number, AuxQubitNumber + 1, oracle, n_solution, is_bit_flip=True)
    result_str = bin(result)[2:].rjust(variable_number,'0')
    is_solution = check_solution([int(x) for x in result_str], variable_number, clause_number, CNF_data)
    print(f"{result_str}, {is_solution}")
    return is_solution

n_all = 100
n_hit = 0
for _ in range(n_all):
    is_solution = test()
    if is_solution:
        n_hit += 1
print(f"success rate: {n_hit/n_all:5.3f}[{n_hit:3}/{n_all:3}]")