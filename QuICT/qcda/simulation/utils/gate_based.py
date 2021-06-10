import numba
import numpy as np
from collections import defaultdict

from QuICT.core.gate.gate import *


# Static gate matrix
STATIC_GATE_NAMES = [
    "HGate", "SGate", "SDaggerGate",
    "XGate", "YGate", "ZGate",
    "SXGate", "SYGate", "SWGate",
    "IDGate", "U1Gate", "U2Gate", "U3Gate"
]

class GateMatrixs:
    def __init__(self, GPUBased: bool=True):
        self.gate_matrixs = {}
        self.GPUBased = GPUBased
        self.matrix_idx = []
        self.matrix_len = 0

    def build(self, gate):
        gate_name = gate.name.split("_")[0]
        if gate_name not in STATIC_GATE_NAMES:
            gate_name = f"{gate_name}_{gate.parg}"
            matrix = gate.compute_matrix
        else:
            matrix = gate.matrix
        
        if gate_name not in self.gate_matrixs.keys():
            self._build_matrix_gate(gate_name, matrix)

    def _build_matrix_gate(self, gate_name, matrix):
        self.gate_matrixs[gate_name] = (self.matrix_len, matrix.size)
        self.matrix_len += matrix.size
        self.matrix_idx.append(matrix)

    def concentrate_gate_matrixs(self):
        self.final_matrix = np.empty(self.matrix_len, dtype=np.complex64)
        start = 0

        for matrix in self.matrix_idx:
            self.final_matrix[start:start+matrix.size] = matrix.ravel()[:]
            start += matrix.size

        if self.GPUBased:
            self.final_matrix = numba.cuda.to_device(self.final_matrix)

    def target_matrix(self, gate):
        gate_name = gate.name.split("_")[0]
        if gate_name not in STATIC_GATE_NAMES:
            gate_name = f"{gate_name}_{gate.parg}"

        start, itvl = self.gate_matrixs[gate_name]

        return self.final_matrix[start:start+itvl]

    @property
    def matrix(self):
        return self.final_matrix
