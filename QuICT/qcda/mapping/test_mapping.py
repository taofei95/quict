import os
import sys
from table_based_mcts import TableBasedMCTS

from typing import List, Tuple, Optional, Union, Dict

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *

def get_line_topology(n: int)->List[Tuple[int,int]]:
    topology:List[Tuple[int,int]] = []
    for i in range(n-1):
        topology.append((i,i+1))
    return topology

small_benchmark = ("rd84_142",
                   "adr4_197",
                   "radd_250",
                   "z4_268",
                   "sym6_145",
                   "misex1_241",
                   "rd73_252",
                   "cycle10_2_110",
                   "square_root_7",
                   "sqn_258",
                   "rd84_253") 

topology = [(0,1),(0,5),
            (1,2),(1,6),(1,7),
            (2,3),(2,6),(2,7),
            (3,4),(3,8),(3,9),
            (4,8),(4,9),
            (5,6),(5,10),(5,11),
            (6,7),(6,10),(6,11),
            (7,8),(7,12),(7,13),
            (8,9),(8,12),(8,13),
            (9,14),
            (10,11),(10,15),
            (11,12),(11,16),(11,17),
            (12,13),(12,16),(12,17),
            (13,14),(13,18),(13,19),
            (14,18),(14,19),
            (15,16),
            (16,17),
            (17,18),
            (18,19)]

def count_two_qubit_gates(circuit: Circuit)->int:
    res = 0
    for gate in circuit.gates:
        if not gate.is_single():
            res = res + 1
    return res

def test_mapping(input_path: str, output_path: str, log_path: str,  num_of_qubits: int, init_mapping: List[int], topology):
    qc = OPENQASMInterface.load_file(input_path)
    circuit =qc.circuit
    circuit_trans = Circuit(wires = 20)
    logical_qubit_num = qc.qbits
    physical_qubit_num = num_of_qubits

    print(input_path)
    mcts = TableBasedMCTS(coupling_graph =  "ibmq20", log_path = log_path)
    mcts.search(logical_circuit = circuit, init_mapping = init_mapping)
    circuit_trans.extend(mcts.physical_circuit)

    print(circuit.circuit_size())
    print(circuit_trans.circuit_size())
    with open(output_path, "w") as f:
        print(circuit.circuit_size(), file = f)
        print(circuit_trans.circuit_size(), file = f)
        print(count_two_qubit_gates(circuit), file = f)
        print(count_two_qubit_gates(circuit_trans), file = f)
        print("Initial layout(physical qubits -> logic qubits):", file = f)

        for i,elm in enumerate(init_mapping):
            print("%d -> %d"%(i, elm), file = f)

        for gate in circuit_trans.gates:
            if gate.controls + gate.targets == 2:
                if gate.controls == 1:
                    print("control:%d  target :%d  gate type:%d" % (gate.carg, gate.targ, gate.type()), file = f)
                elif gate.controls == 2:
                    print("control:%d  target :%d  gate type:%d" % (gate.cargs[0], gate.cargs[1], gate.type()), file = f)
                else:
                    print("control:%d  target :%d  gate type:%d" % (gate.targs[0], gate.targs[1], gate.type()), file = f)
            elif gate.controls + gate.targets == 1:
                print("target :%d  gate type:%d" % (gate.targ, gate.type()), file = f)
        print("——————————————————————————————")

if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    dir_path, file_name = os.path.split(file_path)
    for file_name in small_benchmark:
        input_path = f"{dir_path}/benchmark/QASM example/{file_name}.qasm"
        output_path =  f"{dir_path}/benchmark/QASM example/output/{file_name}.test.output"
        log_path = f"{dir_path}/benchmark/QASM example/output/{file_name}.test.log"
        test_mapping(input_path, output_path, log_path, topology = topology, num_of_qubits = 20, init_mapping = [i for i in range(20)] )