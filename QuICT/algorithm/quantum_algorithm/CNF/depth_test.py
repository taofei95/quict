import os
import time
import numpy as np
import cupy as cp
import math
import random
from QuICT.algorithm.quantum_algorithm.CNF.cnfdepth import *#CNFSATOracle
from QuICT.qcda.optimization import *
from QuICT.core import *
from QuICT.core.gate import *
# from QuICT.simulation.unitary_simulator import UnitarySimulator
from QuICT.simulation.state_vector import ConstantStateVectorSimulator



def test():
    # x0 x1，x2, x_{n variable_number -1}
    filename_test =  "./test_data/4_400_0"
    print(400, "dep")
    #variable_number , clause_number , CNF_data = read_CNF(filename_test)

    #真值表初值变化
    #b=[]
    # print(variable_number)
    #for j in [400,800,1200,1600,2000,2800,3200,3600,4000,7943]:
    for j in [200,300,400,500,600,700,800,1600,2400,3200,4000,4800,5600,6400,7200,8000,8800,9600,10400,11200,12000,12800,13600,14400,15200,15887]:
    #for j in [100,150,200,250,300,350,400,800,1200,1600,2000,2400,2800,3200,3600,4000,4400,4800,5200,5600,6000,6400,6800,7200,7600,7943]:
    #for j in [20,30,40 ,50,60 ,70,80 ,90, 100, 110, 19,18,17,16,15]:
        cnf = CNFSATDEPTHOracle()
        AuxQubitNumber = j
        #print(4 / math.log( j), math.log(4000, j/4 ), 4 * 4000 / j)
        #cnf = CNFSATOracle()
        cnf.run(filename_test, AuxQubitNumber)
        cgate = cnf.circuit()
        #print(cgate.size())
        cgate.gate_decomposition()
        print(cgate.depth())
    #j, cgate.size(), 
    
test()