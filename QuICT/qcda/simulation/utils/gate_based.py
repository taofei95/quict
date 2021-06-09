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
        self.static_gate_matrixs = {}
        self.param_gate_matrixs = defaultdict(dict)
        self.GPUBased = GPUBased
        self.matrix_idx = []
        self.matrix_len = 0

    def build(self, gate):
        gate_name = gate.name.split("_")[0]
        if gate_name in STATIC_GATE_NAMES:
            if gate_name not in self.static_gate_matrixs.keys():
                self._build_static_matrix_gate(gate_name, gate.matrix)
        else:
            if gate.parg not in self.param_gate_matrixs[gate_name].keys():
                self._build_para_matrix_gate(gate_name, gate.parg, gate.compute_matrix)
    
    def _build_static_matrix_gate(self, gate_name, matrix):
        self.static_gate_matrixs[gate_name] = (self.matrix_len, matrix.size)
        self.matrix_len += matrix.size
        self.matrix_idx.append(matrix)

    def _build_para_matrix_gate(self, gate_name, gate_parg, matrix):
        self.param_gate_matrixs[gate_name][gate_parg] = (self.matrix_len, matrix.size)
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
        if gate_name in STATIC_GATE_NAMES:
            return self.static_gate_matrixs[gate_name]
        else:
            gate_parg = gate.parg
            return self.param_gate_matrixs[gate_name][gate_parg]

    @property
    def matrix(self):
        return self.final_matrix


class GateMatrixsMS:
    def __init__(self, streams_num: int):
        self.static_gate_matrixs = {}
        self.param_gate_matrixs = defaultdict(dict)

        self.stream_num = streams_num
        self.streams = []
        for i in range(self.stream_num):
            self.streams.append(numba.cuda.stream())

        self.count = 0

    def build(self, gate):
        gate_name = gate.name.split("_")[0]
        if gate_name in STATIC_GATE_NAMES:
            if gate_name not in self.static_gate_matrixs.keys():
                self._build_static_matrix_gate(gate_name, gate.matrix)
        else:
            if gate.parg not in self.param_gate_matrixs[gate_name].keys():
                self._build_para_matrix_gate(gate_name, gate.parg, gate.compute_matrix)

    def _build_static_matrix_gate(self, gate_name, matrix):
        self.static_gate_matrixs[gate_name] = numba.cuda.to_device(matrix, self.streams[self.count%self.stream_num])
        self.count += 1

    def _build_para_matrix_gate(self, gate_name, gate_parg, matrix):
        self.param_gate_matrixs[gate_name][gate_parg] = numba.cuda.to_device(matrix, self.streams[self.count%self.stream_num])
        self.count += 1

    def target_matrix(self, gate):
        gate_name = gate.name.split("_")[0]
        if gate_name in STATIC_GATE_NAMES:
            return self.static_gate_matrixs[gate_name]
        else:
            gate_parg = gate.parg
            return self.param_gate_matrixs[gate_name][gate_parg]
