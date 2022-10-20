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

def test():
    filename_test =  "./test_data/3_6_100"
    AuxQubitNumber = 5
    variable_number , clause_number , CNF_data = read_CNF(filename_test)
    #print(CNF_data)
    #真值表初值变化
    #b=[]
    # print(variable_number)
    cnf = CNFSATOracle()
    cnf.run(filename_test, AuxQubitNumber,1)
    oracle = cnf.circuit()
    grover = Grover(simulator=TestGrover.simulator)
    
    result = grover.run(variable_number, AuxQubitNumber, oracle)


test()