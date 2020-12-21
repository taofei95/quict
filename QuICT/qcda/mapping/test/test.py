import os
import sys

file_path = os.path.realpath(__file__)
dir_path, file_name = os.path.split(file_path)


from QuICT.qcda.mapping import Mapping
from QuICT.tools.interface import *

topology = [(0,1),(1,2),(2,3),(),()  ]


if __name__ == "__main__":

    QASM_file = f"{dir_path}/benchmark/qasm/qft/qft_n6.qasm"
    qc = OPENQASMInterface.load_file(QASM_file)
    circuit =qc.circuit
    circuit.add_topology()
    num = qc.qbits
    #print(num)
    init_mapping = [i for i in range(num)]

    circuit_trans = Mapping.run(circuit = circuit,num =  num, method = "global_sifting", is_lnn = True,init_mapping= init_mapping)


    print(circuit.circuit_size())
    print(circuit_trans.circuit_size())
    print("Initial layout(physical qubits -> logic qubits):")
  
    for i,elm in enumerate(init_mapping):
        print("%d -> %d"%(i, elm))

    for gate in circuit_trans.gates:
        if gate.controls + gate.targets == 2:
            if gate.controls == 1:
                print("control:%d  target :%d  gate type:%d" % (gate.carg, gate.targ, gate.type().value))
            elif gate.controls == 2:
                print("control:%d  target :%d  gate type:%d" % (gate.cargs[0], gate.cargs[1], gate.type().value))
            else:
                print("control:%d  target :%d  gate type:%d" % (gate.targs[0], gate.targs[1], gate.type().value))
        elif gate.controls + gate.targets == 1:
            print("target :%d  gate type:%d" % (gate.targ, gate.type().value))