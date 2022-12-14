
import os
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_algorithm.grover import Grover
from QuICT.algorithm.quantum_algorithm.CNF.cnf import CNFSATOracle


def read_CNF(cnf_file):
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
            CNF_data.append(int_new)
        f.close()
        return variable_number, clause_number, CNF_data
    
def check_solution(variable_data, variable_number, clause_number, CNF_data):
    cnf_result = 1
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

path = 'QuICT/algorithm/quantum_algorithm/CNF/1129/2_3_1'
# files_2 = os.listdir(path+'/1129')

variable_number, clause_number, CNF_data = read_CNF(path)
ancilla_qubits_num=3
dirty_ancilla=1
cnf = CNFSATOracle()
cnf.run(path, ancilla_qubits_num, dirty_ancilla)

oracle = cnf.circuit()
grover = Grover(ConstantStateVectorSimulator())

circ = grover.circuit(variable_number, ancilla_qubits_num + dirty_ancilla, oracle, n_solution=2, measure=False, is_bit_flip=True, iteration_number_forced=True)
zong = variable_number + oracle.width()*2
print(zong)
print(circ.width(), circ.depth(), circ.size())