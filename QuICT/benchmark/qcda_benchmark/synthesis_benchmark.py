import pandas as pd
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import CLIFFORD_GATE_SET
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.synthesis import (
    UnitaryDecomposition,
    GateTransform,
    CliffordUnidirectionalSynthesizer,
    QuantumStatePreparation
)
from scipy.stats import unitary_group

def gate_count(circuit):
    size = circuit.size()
    depth = circuit.depth()
    
    return size, depth

def unitary_decomposition(qubit_num):
    circuit = Circuit(qubit_num)
    circuit.random_append(20 * qubit_num)
    matrix = unitary_group.rvs(2 ** qubit_num)
    UD = UnitaryDecomposition()
    circuit_opt, _ = UD.execute(matrix)
    return circuit, circuit_opt

def gate_transform(qubit_num):
    circuit = Circuit(qubit_num)
    circuit.random_append(20 * qubit_num)
    GT = GateTransform()
    circuit_opt = GT.execute(circuit)
    return circuit, circuit_opt

def Clifford_synthesizer(qubit_num):
    circuit = Circuit(qubit_num)
    circuit.random_append(20 * qubit_num, CLIFFORD_GATE_SET)
    CUS = CliffordUnidirectionalSynthesizer()
    circuit_opt = CUS.execute(circuit)
    return circuit, circuit_opt

def Quantum_state_preparation(qubit_num):
    circuit = Circuit(qubit_num)
    circuit.random_append(20 * qubit_num)
    matrix = unitary_group.rvs(2 ** qubit_num)
    TO = QuantumStatePreparation()
    circuit_opt = TO.execute(matrix)
    return circuit, circuit_opt


if __name__ == '__main__':
    data = open('synthesis_benchmark_data.txt', 'w+')
    qubit_number = [5, 10, 15, 20]
    opt_function = [unitary_decomposition, gate_transform, Clifford_synthesizer, Quantum_state_preparation]
    opt_function_name = ["unitary_decomposition", "gate_transform", "Clifford_synthesizer", "Quantum_state_preparation"]
    for j in opt_function:
        data.write(f"functon:{opt_function_name[opt_function.index(j)]} \n")
        for i in qubit_number:
            data.write(f"qubit number:{i} \n")
            data.write(f"before opt:{gate_count(j(i)[0])} \n")
            data.write(f"after opt:{gate_count(j(i)[1])} \n")

    data.close()



