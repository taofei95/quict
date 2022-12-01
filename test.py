
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
    
# for i in range(3, 98):
filename = f"2_4_1"
filename_test =  "QuICT/algorithm/quantum_algorithm/CNF/1129/" + filename
AuxQubitNumber = 3
variable_number , clause_number , CNF_data = read_CNF(filename_test)

cnf = CNFSATOracle()
cnf.run(filename_test, AuxQubitNumber)
cgate = cnf.circuit()

folder_path = "QuICT/algorithm/quantum_algorithm/CNF/1201/cnf"
file = open(folder_path + '/' + f"w{cgate.width()}_s{cgate.size()}_d{cgate.depth()}.qasm",'w+')
file.write(cgate.qasm())
file.close()