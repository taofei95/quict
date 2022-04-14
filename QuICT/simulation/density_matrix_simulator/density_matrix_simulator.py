from operator import matmul
from unittest.result import failfast
import numpy as np
import math
import random
from QuICT.core.circuit.circuit import Circuit
from QuICT.simulation.unitary_simulator import UnitarySimulator
import QuICT.ops.linalg.cpu_calculator as CPUCalculator
from QuICT.core.gate import Measure, BasicGate
from QuICT.core.utils import GateType
from QuICT.core.utils import matrix_product_to_circuit

# TODO: Remove unused import

class DensityMatrixSimulation:

    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double"
    ):
        self._device = device
        self._precision = precision

        if device == "CPU":
            self._computer = CPUCalculator
        else:
            import QuICT.ops.linalg.gpu_calculator as GPUCalculator
            self._computer = GPUCalculator


    def _init_density_matrix(self, qubits):
        # TODO: Generate density matrix depending on device, follow the _initial_vector_state in UnitarySimulator
        if self._device == "CPU":
            self._density_matrix = np.zeros((1 << qubits, 1 << qubits), dtype = self._precision)
            self._density_matrix[0,0] = self._precision(1)
        else:
            import cupy as cp
            
            self._density_matrix = cp.zeros((1 << qubits, 1 << qubits), dtype = self._precision)
            self._density_matrix.put((0, 0), self._precision(1))
            # ?

    def check_matrix(matrix):
        if(matrix.T.conjugate() != matrix): 
            return False

        eigenvalues = np.linalg.eig(matrix)[0]
        for ev in eigenvalues:
            if(ev < 0):
                return False

        if(matrix.trace() != 1):
            return False

        return True

    def _measure(self, gate_list, qubits):
        # TODO: use np.array(matrix, dtype) to generate P0, generate P1 in condition "else"
        P0 = np.array([[1, 0], [0, 0]], dtype = self._precision)
        # P0 = np.mat([[1, 0], [0, 0]]) + "target qubits" ?

        # TODO: Using matrix_product_to_circuit; from QuICT.core.utils import matrix_product_to_circuit
        mea_0 = matrix_product_to_circuit(P0)
        # TODO: no matmul in computer currently, use np.matmul
        # TODO: matrix == self._density_matrix
        prob_0 = self.np.matmul(mea_0, self._density_matrix).trace()
        _0_1 = random() < prob_0
        if _0_1:
            # TODO: n = qubits
            U = self.np.matmul(mea_0, np.eye(1 << qubits) / np.sqrt(prob_0))
            # TODO: use self._computer.dot here
            self._density_matrix = self._computer.dot(self._computer.dot(U, self._density_matrix), U.conj().T)
            # matrix = U.dot(matrix).dot(U.conj().T)
        else:
            P1 = np.array([[0, 0], [0, 1]], dtype = self._precision)
            mea_1 = matrix_product_to_circuit(P1)
            U = np.matmul(mea_1, np.eye(1 << qubits)/np.sqrt(1 - prob_0))
            self._density_matrix = self._computer.dot(self._computer.dot(U, self._density_matrix), U.conj().T)
            # matrix = U.dot(self._density_matrix).dot(U.conj().T)

        # TODO: return _0_1, to change qubit in density_matrix_simu function; due to no circuit here.
        return _0_1

    # TODO: Rename to run
    def _run(self, circuit:Circuit, qubits, density_matrix: np.ndarray = None):
        if(density_matrix == None or self.check_matrix(density_matrix) == False):
            # TODO: no return here
            self._init_density_matrix(self, qubits)
        else:
            self._density_matrix = density_matrix
        
        # Assume no measure gate in circuit middle, measure gate only appear last
        # circuit.gates [non-measure gates] [measure gate]
        measure_gate_list = []
        for gate in circuit.gates:
            if gate.type == GateType.measure:
                measure_gate_list.append(gate) 
                circuit.gates.remove(gate)  
        circuit_matrix = UnitarySimulator.get_unitary_matrix(circuit)
        
        # step ops
        # TODO: using self.density_matrix
        self._density_matrix = self._computer.dot(
            self._computer.dot(circuit_matrix, self._density_matrix),
            circuit_matrix.conj().T
        )

        # [measure gate] exist
        # TODO: add if measure_gate_list: do Measure and change circuit.qubits.measured
        if measure_gate_list:
            _0_1 = self._measure(measure_gate_list, circuit.width())
            circuit.qubits[gate.targ].measured = int(_0_1)
            # ?

        return self._density_matrix


