import os
import numpy as np
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from qiskit import QuantumCircuit, transpile, Aer, QuantumRegister
from scipy.stats import unitary_group
from scipy.stats import unitary_group
from QuICT.core.gate.gate import Unitary
from QuICT.qcda.synthesis.unitary_decomposition.unitary_decomposition import UnitaryDecomposition
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

f = open("unitary_composition_synthesis_benchmark_data.txt", 'w+')

qubit_num = [3, 4, 5, 6, 7, 8, 9, 10]
for i in qubit_num:
    matrix = unitary_group.rvs(2 ** i)
    cir, _ = UnitaryDecomposition().execute(matrix)

    f.write(f"Quict circuit size:{cir.size()}, Quict circuit depth:{cir.depth()}\n")

    q_i = QuantumRegister(i)
    qcir = QuantumCircuit(q_i)
    qcir.isometry(
        isometry = matrix,
        q_input =q_i,
        q_ancillas_for_output = []
    )

    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(qcir, simulator)

    f.write(f"Qiskit circuit size:{circ.size()}, Qiskit circuit depth:{circ.depth()}\n")

f.close()