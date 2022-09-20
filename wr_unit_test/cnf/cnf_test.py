import os
import time
import numpy as np
from QuICT.algorithm.quantum_algorithm.CNF.cnf import CNFSATOracle, read_CNF
from QuICT.qcda.optimization import *
# from QuICT.simulation.unitary_simulator import UnitarySimulator
from QuICT.simulation.cpu_simulator import CircuitSimulator

cnf = CNFSATOracle()
cnf.run("./wr_unit_test/cnf/cnf_test_data/3_11_100")
cgate = cnf.circuit()

#get q(n)
sim = CircuitSimulator()
amplitude = sim.run(cgate)
print(amplitude[-1])


def test(
        self, 
        variable_nunmber : int, 
        clause_number : int, 
        CNF_data = read_CNF("./wr_unit_test/cnf/cnf_test_data/3_11_100")
        ):
# x0 x1，x2, x_{n variable_nunmber -1}
    for i in range(variable_nunmber):
        x[i]=0
    cnf_result = 1
    for i in range(clause_number):
        clause_result = 0
        for j in range(CNF_data[i+1]):
            if CNF_data[i+1][j] > 0:
                clause_result += x[CNF_data[i+1][j]-1] 
                else:
                    clause_result += (1 - x[-CNF_data[i+1][j]-1] )
            if clause_result == 0:
                cnf_result = 0
                break
test()