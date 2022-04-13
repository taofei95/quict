import abc
from operator import matmul
from statistics import mean
from tabnanny import check
from unittest.result import failfast
from venv import create
import numpy as np
import pandas
import math
import random
from QuICT.core.circuit.circuit import Circuit
from QuICT.simulation.unitary_simulator import UnitarySimulator
import QuICT.ops.linalg.cpu_calculator as CPUCalculator
import QuICT.core.utils
from QuICT.core.gate import Measure, BasicGate
from QuICT.core.utils import GateType

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


    def create_init_density_matrix(self, n):
        self.density_matrix = np.zeros((2 ** n, 2 ** n), dtype = self._precision)
        self.density_matrix[0,0] = 1

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

    def measure(self, gate_list, qubits):
        
        P0 = np.mat([[1, 0], [0, 0]]) + "target qubits"
        P1 = np.mat([[0, 0], [0, 1]])
        mea_0 = mea_circuit.matrix_product_to_circuit(P0)
        prob_0 = self._computer.matmul(mea_0, matrix).trace()
        _0_1 = random() < prob_0
        if _0_1:
            U = self._computer.matmul(mea_0, np.eye(2**n) / np.sqrt(prob_0))
            matrix = U.dot(matrix).dot(U.conj().T)
        else:
            mea_1 = mea_circuit.matrix_product_to_circuit(P1)
            U = np.matmul(mea_1, np.eye(2**n)/np.sqrt(1 - prob_0))
            matrix = U.dot(matrix).dot(U.conj().T)

        mea_circuit.qubits[gate.targ].measured = int(_0_1)


    def density_matrix_simu(self, circuit:Circuit, n, density_matrix: np.ndarray = None):
        if(density_matrix == None or self.check_matrix(density_matrix) == False):
            density_matrix = self.create_init_density_matrix(n)
        else:
            self.density_matrix = density_matrix
        
        # Assume no measure gate in circuit middle, measure gate only appear last
        # circuit.gates [non-measure gates] [measure gate]
        measure_gate_list = []
        for gate in circuit.gates:
            if gate.type == GateType.measure:
                measure_gate_list.append(gate) 
                circuit.gates.remove(gate)  
        circuit_matrix = UnitarySimulator.get_unitary_matrix(circuit)
        
        # step ops
        self.density_matrix = self._computer.dot(self._computer.dot(circuit_matrix, density_matrix), circuit_matrix.conj().T)

        # [measure gate] exist
        self._measure(measure_gate_list, circuit.width())

        return density_matrix


