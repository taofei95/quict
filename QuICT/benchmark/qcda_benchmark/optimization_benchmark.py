import pandas as pd
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import CLIFFORD_GATE_SET
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.optimization import (
    CliffordRzOptimization,
    CommutativeOptimization,
    SymbolicCliffordOptimization,
    TemplateOptimization
)

def gate_count(circuit):
    size = circuit.size()
    depth = circuit.depth()
    cx_num = circuit.count_gate_by_gatetype(GateType.cx)
    return size, depth, cx_num

def build_clifford(qubit_num):
    circuit = Circuit(qubit_num)
    circuit.random_append(20 * qubit_num, CLIFFORD_GATE_SET)
    return circuit

def build_random_circuit(qubit_num):
    circuit = Circuit(qubit_num)
    circuit.random_append(20 * qubit_num)
    return circuit

def Symbolic_Clifford_Optimization(qubit_num):
    circuit = build_clifford(qubit_num)
    SCO = SymbolicCliffordOptimization()
    circuit_opt = SCO.execute(circuit)
    return circuit, circuit_opt

def Clifford_Rz_Optimization(qubit_num):
    circuit = build_random_circuit(qubit_num)
    CRO = CliffordRzOptimization()
    circuit_opt = CRO.execute(circuit)
    return circuit, circuit_opt

def Commutative_Optimization(qubit_num):
    circuit = build_random_circuit(qubit_num)
    CO = CommutativeOptimization()
    circuit_opt = CO.execute(circuit)
    return circuit, circuit_opt

def Template_Optimization(qubit_num):
    circuit = build_clifford(qubit_num)
    TO = TemplateOptimization()
    circuit_opt = TO.execute(circuit)
    return circuit, circuit_opt


if __name__ == '__main__':
    data = open('optimization_benchmark_data.txt', 'w+')
    qubit_number = [5, 10, 15, 20]
    opt_function = [Symbolic_Clifford_Optimization, Clifford_Rz_Optimization, Commutative_Optimization, Template_Optimization]
    opt_function_name = ["Symbolic_Clifford_Optimization", "Clifford_Rz_Optimization", "Commutative_Optimization", "Template_Optimization"]
    for j in opt_function:
        data.write(f"functon:{opt_function_name[opt_function.index(j)]} \n")
        for i in qubit_number:
            data.write(f"qubit number:{i} \n")
            data.write(f"before opt:{gate_count(j(i)[0])} \n")
            data.write(f"after opt:{gate_count(j(i)[1])} \n")

    data.close()



