#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/9 下午5:35
# @Author  : Kaiqi Li
# @File    : gate_based

import numpy as np
import cupy as cp


_GATES_EXCEPT = ["MeasureGate", "ResetGate", "PermFxGate", "PermGate"]


class GateMatrixs:
    """
    The class of storing the gates' compute matrix, without duplicate.

    Args:
        precision(Union[np.complex64, np.complex128]): The precision of the gates.
        gpu_device_id(int): The GPU device ID.
    """
    def __init__(self, precision, gpu_device_id: int = 0):
        self.gate_matrixs = {}
        self.precision = precision
        self.device_id = gpu_device_id
        self.matrix_idx = []
        self.matrix_len = 0

    @property
    def matrix(self):
        """ Return the matrix with all gates' compute matrix. """
        return self.final_matrix

    def build(self, gate):
        """
        Add gate into GateMatrixs, if the gate is the new one.

        Args:
            gate(Gate): the gate in circuit.
        """
        gate_name = gate.name.split("_")[0]
        if gate_name in _GATES_EXCEPT:
            return

        param_num = gate.params
        if gate.params != 0:
            for i in range(param_num):
                gate_name += f"_{gate.pargs[i]}"

        if gate_name == "UnitaryGate":
            gate_name = gate.name

        if gate_name not in self.gate_matrixs.keys():
            matrix = gate.compute_matrix
            self._build_matrix_gate(gate_name, matrix)

    def _build_matrix_gate(self, gate_name, matrix):
        """ Add gate. """
        self.gate_matrixs[gate_name] = (self.matrix_len, matrix.size)
        self.matrix_len += matrix.size
        if matrix.dtype != self.precision:
            matrix = matrix.astype(self.precision)
        self.matrix_idx.append(matrix)

    def concentrate_gate_matrixs(self):
        """ Combined all gates' computer matrix in one large matrix."""
        self.final_matrix = np.empty(self.matrix_len, dtype=self.precision)
        start = 0

        for matrix in self.matrix_idx:
            self.final_matrix[start:start + matrix.size] = matrix.ravel()[:]
            start += matrix.size

        with cp.cuda.Device(self.device_id):
            self.final_matrix = cp.array(self.final_matrix)

    def target_matrix(self, gate):
        """
        Find the compute matrix of the given gate.
        Args:
            gate(Gate): the gate in circuit.
        """
        gate_name = gate.name.split("_")[0]
        param_num = gate.params
        if param_num != 0:
            for i in range(param_num):
                gate_name += f"_{gate.pargs[i]}"

        if gate_name in _GATES_EXCEPT:
            raise KeyError(f"Wrong gate here. {gate_name}")

        if gate_name == "UnitaryGate":
            gate_name = gate.name

        start, itvl = self.gate_matrixs[gate_name]

        return self.final_matrix[start:start + itvl]
