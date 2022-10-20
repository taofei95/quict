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
    Grover,
    PartialGrover,
    GroverWithPriorKnowledge,
)
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux

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