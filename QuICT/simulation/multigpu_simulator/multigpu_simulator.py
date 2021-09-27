#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/30 下午5:50
# @Author  : Kaiqi Li
# @File    : proxy_simulator

import numpy as np
import cupy as cp
import os

from QuICT.core import *
from QuICT.simulation import BasicGPUSimulator
from QuICT.ops.utils import Proxy, LinAlgLoader, perm_sort
from QuICT.simulation.utils import GateType, GATE_TYPE_to_ID, MATRIX_INDEXES
from QuICT.simulation.multigpu_simulator.data_switch import DataSwitcher


LIMIT_BUFFER_SIZE = int(os.getenv("QuICT_BUFFER_SIZE", 17))


class MultiStateVectorSimulator(BasicGPUSimulator):
    """
    The simulator which using multi-GPUs.

    Args:
        proxy (Proxy): The NCCL communicators.
        circuit (Circuit): The quantum circuit.
        precision [np.complex64, np.complex128]: The precision for the circuit and qubits.
        gpu_device_id (int): The GPU device ID.
        sync (bool): Sync mode or Async mode.
    """
    def __init__(
        self,
        proxy: Proxy,
        circuit: Circuit,
        precision=np.complex64,
        gpu_device_id: int = 0,
        sync: bool = True
    ):
        self.proxy = proxy
        self._sync = sync
        self._buffer_size = LIMIT_BUFFER_SIZE if precision == np.complex64 else LIMIT_BUFFER_SIZE - 1
        assert(proxy.rank == gpu_device_id)

        # Initial simulator with qubits
        BasicGPUSimulator.__init__(self, circuit, precision, gpu_device_id)

        # Get qubits and limitation
        self.total_qubits = int(circuit.circuit_width())
        self.qubits = int(self.total_qubits - np.log2(self.proxy.ndevs))

        self.initial_vector_state()

        # Initial the required algorithm.
        self._algorithm = LinAlgLoader(device="GPU", enable_gate_kernel=True, enable_multigpu_gate_kernel=True)
        self._data_switcher = DataSwitcher(self.proxy, self.qubits)

    def initial_vector_state(self):
        """
        Initial qubits' vector states.
        """
        vector_size = 1 << int(self.qubits)
        # Special Case for no gate circuit
        if len(self._gates) == 0:
            self._vector = np.zeros(vector_size, dtype=self._precision)
            if self.proxy.rank == 0:
                self._vector[0] = self._precision(1)
            return

        # Initial qubit's states
        with cp.cuda.Device(self._device_id):
            self._vector = cp.empty(vector_size, dtype=self._precision)
            if self.proxy.rank == 0:
                self._vector.put(0, self._precision(1))

    def run(self) -> np.ndarray:
        """
        Start simulator.
        """
        with cp.cuda.Device(self._device_id):
            for gate in self._gates:
                self.apply_gate(gate)

        return self.vector

    def apply_gate(self, gate):
        """
        Trigger Gate event in the circuit.
        """
        gate_type = gate.type()
        is_switch_data = False
        if gate_type in GATE_TYPE_to_ID[GateType.matrix_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            if t_index >= self.qubits:
                destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                self._data_switcher.half_switch(self._vector, destination)
                is_switch_data = True
                t_index = self.qubits - 1

            self._algorithm.Based_InnerProduct_targ(
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )

            if is_switch_data:
                self._data_switcher.half_switch(self._vector, destination)
        elif gate_type in GATE_TYPE_to_ID[GateType.diagonal_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            if t_index >= self.qubits:
                index = 3 if self.proxy.rank & (1 << (t_index - self.qubits)) else \
                    0

                self._algorithm.Simple_Multiply(
                    index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
            else:
                self._algorithm.Diagonal_Multiply_targ(
                    t_index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            if t_index >= self.qubits:
                destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                self._data_switcher.all_switch(self.vector, destination)
            else:
                self._algorithm.RDiagonal_Swap_targ(
                    t_index,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            if t_index >= self.qubits:
                destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                index = 1 if self.proxy.rank & (1 << (t_index - self.qubits)) else \
                    2

                self._algorithm.Simple_Multiply(
                    index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )

                self._data_switcher.all_switch(self._vector, destination)
            else:
                self._algorithm.RDiagonal_MultiplySwap_targ(
                    t_index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.control_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            if (
                t_index >= self.qubits and
                self.proxy.rank & (1 << (t_index - self.qubits))
            ):
                self._algorithm.Simple_Multiply(
                    3,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
            else:
                self._algorithm.Controlled_Multiply_targ(
                    t_index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.control_2arg]:
            t_index = self.total_qubits - 1 - gate.targ
            c_index = self.total_qubits - 1 - gate.carg
            matrix = self.get_gate_matrix(gate)
            if t_index >= self.qubits and c_index >= self.qubits:
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    index = 15 if self.proxy.rank & (1 << (t_index - self.qubits)) else \
                        10
                    self._algorithm.Simple_Multiply(
                        index,
                        matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            elif c_index >= self.qubits:
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    self._algorithm.Diagonal_Multiply_targ(
                        t_index,
                        matrix[MATRIX_INDEXES[0]],
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            elif t_index >= self.qubits:
                temp_matrix = cp.array([1, 0, 0, 0], dtype=self._precision)
                temp_matrix[3] = matrix[15] if self.proxy.rank & (1 << (t_index - self.qubits)) else \
                    matrix[10]

                self._algorithm.Controlled_Multiply_targ(
                    c_index,
                    temp_matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
            else:
                position = 12 if gate_type == GATE_ID["CRz"] else 8
                self._algorithm.Controlled_Multiply_ctargs(
                    c_index,
                    t_index,
                    matrix,
                    self._vector,
                    self._qubits,
                    position,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.diagonal_2arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            if t_indexes[0] > t_indexes[1]:
                t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

            if t_indexes[0] >= self.qubits:
                _0 = self.proxy.rank & (1 << (t_indexes[0] - self.qubits))
                _1 = self.proxy.rank & (1 << (t_indexes[1] - self.qubits))

                index = 0
                if _0:
                    index += 5
                if _1:
                    index += 10

                self._algorithm.Simple_Multiply(
                    index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
            elif t_indexes[1] >= self.qubits:
                temp_matrix = cp.zeros((4,), dtype=self._precision)
                if self.proxy.rank & (1 << (t_indexes[1] - self.qubits)):
                    temp_matrix[0], temp_matrix[3] = matrix[10], matrix[15]
                else:
                    temp_matrix[0], temp_matrix[3] = matrix[0], matrix[5]

                self._algorithm.Diagonal_Multiply_targ(
                    t_indexes[0],
                    temp_matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
            else:
                self._algorithm.Diagonal_Multiply_targs(
                    t_indexes,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_2arg]:
            t_index = self.total_qubits - 1 - gate.targ
            c_index = self.total_qubits - 1 - gate.carg
            matrix = self.get_gate_matrix(gate)
            if t_index >= self.qubits and c_index >= self.qubits:
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    index = 11 if self.proxy.rank & (1 << (t_index - self.qubits)) else \
                        14
                    self.Simple_Multiply(
                        index,
                        matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )

                    destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                    self._data_switcher.all_switch(self._vector, destination)
            elif t_index >= self.qubits:
                temp_matrix = cp.zeros((4,), dtype=self._precision)
                temp_matrix[0] = 1
                temp_matrix[3] = matrix[11] if self.proxy.rank & (1 << (t_index - self.qubits)) else \
                    matrix[14]

                self._algorithm.Controlled_Multiply_targ(
                    c_index,
                    temp_matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )

                destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    {c_index: 1}
                )
            elif c_index >= self.qubits:
                temp_matrix = cp.zeros((4,), dtype=self._precision)
                temp_matrix[1], temp_matrix[2] = matrix[11], matrix[14]
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    self._algorithm.RDiagonal_MultiplySwap_targ(
                        t_index,
                        temp_matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            else:
                self._algorithm.Controlled_MultiplySwap_ctargs(
                    c_index,
                    t_index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.matrix_2arg]:
            t_index = self.total_qubits - 1 - gate.targ
            c_index = self.total_qubits - 1 - gate.carg
            matrix = self.get_gate_matrix(gate)
            if t_index >= self.qubits and c_index >= self.qubits:
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                    self._data_switcher.half_switch(
                        self._vector,
                        destination
                    )

                    self._algorithm.Based_InnerProduct_targ(
                        self.qubits - 1,
                        matrix[MATRIX_INDEXES[0]],
                        self._vector,
                        self._qubits,
                        self._sync
                    )

                    self._data_switcher.half_switch(
                        self._vector,
                        destination
                    )
            elif c_index >= self.qubits:
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[0]]
                    self._algorithm.Based_InnerProduct_targ(
                        t_index,
                        temp_matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            elif t_index >= self.qubits:
                destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                switch_condition = {c_index: int(self.proxy.rank < destination)}

                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    switch_condition
                )

                if self.proxy.rank & (1 << (t_index - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[0]]
                    self._algorithm.Based_InnerProduct_targ(
                        c_index,
                        temp_matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )

                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    switch_condition
                )
            else:
                self._algorithm.Controlled_InnerProduct_ctargs(
                    c_index,
                    t_index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.complexMIP_2arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            if t_indexes[0] > t_indexes[1]:
                t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

            if t_indexes[0] >= self.qubits:
                _0 = self.proxy.rank & (1 << (t_indexes[0] - self.qubits))
                _1 = self.proxy.rank & (1 << (t_indexes[1] - self.qubits))

                if _0 and _1:
                    self._algorithm.Simple_Multiply(
                        15,
                        matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
                elif _0 or _1:
                    destination = self.proxy.rank ^ \
                        ((1 << (t_indexes[0] - self.qubits)) + (1 << (t_indexes[1] - self.qubits)))
                    self._data_switcher.half_switch(
                        self._vector,
                        destination
                    )
                    self._algorithm.Based_InnerProduct_targ(
                        self.qubits - 1,
                        matrix[MATRIX_INDEXES[1]],
                        self._vector,
                        self._qubits,
                        self._sync
                    )
                    self._data_switcher.half_switch(
                        self._vector,
                        destination
                    )
                else:
                    pass
            elif t_indexes[1] >= self.qubits:
                destination = self.proxy.rank ^ (1 << (t_indexes[1] - self.qubits))
                self._data_switcher.ctargs_switch(
                    self.vector,
                    destination,
                    {t_indexes[0]: 1}
                )

                if self.proxy.rank & (1 << (t_indexes[1] - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[2]]
                    self._algorithm.Based_InnerProduct_targ(
                        t_indexes[0],
                        temp_matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
                else:
                    temp_matrix = matrix[MATRIX_INDEXES[3]]
                    self._algorithm.Diagonal_Multiply_targ(
                        t_indexes[0],
                        temp_matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )

                self._data_switcher.ctargs_switch(
                    self.vector,
                    destination,
                    {t_indexes[0]: 1}
                )
            else:
                self._algorithm.Completed_MxIP_targs(
                    t_indexes,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.complexIPIP_2arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            if t_indexes[0] > t_indexes[1]:
                t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

            if t_indexes[0] >= self.qubits:
                destination = self.proxy.rank ^ \
                    ((1 << (t_indexes[0] - self.qubits)) + (1 << (t_indexes[1] - self.qubits)))
                self._data_switcher.half_switch(
                    self._vector,
                    destination
                )

                if (
                    self.proxy.rank & (1 << (t_indexes[0] - self.qubits)) ==
                    self.proxy.rank & (1 << (t_indexes[1] - self.qubits))
                ):
                    temp_matrix = matrix[MATRIX_INDEXES[3]]
                else:
                    temp_matrix = matrix[MATRIX_INDEXES[1]]

                self._algorithm.Based_InnerProduct_targ(
                    self.qubits - 1,
                    temp_matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )

                self._data_switcher.half_switch(
                    self._vector,
                    destination
                )
            elif t_indexes[1] >= self.qubits:
                destination = self.proxy.rank ^ (1 << (t_indexes[1] - self.qubits))
                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    {t_indexes[0]: 1}
                )

                if self.proxy.rank & (1 << (t_indexes[1] - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[2]]
                else:
                    temp_matrix = matrix[MATRIX_INDEXES[3]]

                self._algorithm.Based_InnerProduct_targ(
                    t_indexes[0],
                    temp_matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )

                self._data_switcher.ctargs_switch(
                    self.vector,
                    destination,
                    {t_indexes[0]: 1}
                )
            else:
                self._algorithm.Completed_IPxIP_targs(
                    t_indexes,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_2arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            if t_indexes[0] > t_indexes[1]:
                t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

            self.swap_operation(t_indexes)
        elif gate.type() == GATE_ID["ID"]:
            pass
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_3arg]:
            c_indexes = [self.total_qubits - 1 - carg for carg in gate.cargs]
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            if c_indexes[0] > c_indexes[1]:
                c_indexes[0], c_indexes[1] = c_indexes[1], c_indexes[0]

            _t = t_index >= self.qubits
            _c0 = c_indexes[0] >= self.qubits
            _c1 = c_indexes[1] >= self.qubits
            if _t and _c0:
                if (
                    self.proxy.rank & (1 << (c_indexes[0] - self.qubits)) and
                    self.proxy.rank & (1 << (c_indexes[1] - self.qubits))
                ):
                    destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                    self._data_switcher.all_switch(self._vector, destination)
            elif _t and _c1:
                if self.proxy.rank & (1 << (c_indexes[1] - self.qubits)):
                    destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                    switch_condition = {c_indexes[0]: 1}
                    self._data_switcher.ctargs_switch(
                        self._vector,
                        destination,
                        switch_condition
                    )
            elif _c0:
                if (
                    self.proxy.rank & (1 << (c_indexes[0] - self.qubits)) and
                    self.proxy.rank & (1 << (c_indexes[1] - self.qubits))
                ):
                    self._algorithm.RDiagonal_Swap_targ(
                        t_index,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            elif _c1:
                if self.proxy.rank & (1 << (c_indexes[1] - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[4]]
                    self._algorithm.Controlled_MultiplySwap_ctargs(
                        c_indexes[0],
                        t_index,
                        temp_matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            elif _t:
                destination = self.proxy.rank ^ (1 << (t_index - self.qubits))
                switch_condition = {
                    c_indexes[1]: 1,
                    c_indexes[0]: 1
                }
                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    switch_condition
                )
            else:
                self._algorithm.Controlled_Swap_more(
                    c_indexes,
                    t_index,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.control_3arg]:
            c_indexes = [self.total_qubits - 1 - carg for carg in gate.cargs]
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            if c_indexes[0] > c_indexes[1]:
                c_indexes[0], c_indexes[1] = c_indexes[1], c_indexes[0]

            _t = t_index >= self.qubits
            _c0 = c_indexes[0] >= self.qubits
            _c1 = c_indexes[1] >= self.qubits
            if _t and _c0:
                if (
                    self.proxy.rank & (1 << (c_indexes[0] - self.qubits)) and
                    self.proxy.rank & (1 << (c_indexes[1] - self.qubits))
                ):
                    index = 63 if self.proxy.rank & (1 << (t_index - self.qubits)) else \
                        54

                    self._algorithm.Simple_Multiply(
                        index,
                        matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            elif _t and _c1:
                if self.proxy.rank & (1 << (c_indexes[1] - self.qubits)):
                    if self.proxy.rank & (1 << (c_indexes[1] - self.qubits)):
                        temp_matrix = matrix[MATRIX_INDEXES[5]]
                    else:
                        temp_matrix = matrix[MATRIX_INDEXES[6]]
                    self._algorithm.Controlled_Multiply_targ(
                        c_indexes[0],
                        temp_matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            elif _c0:
                if (
                    self.proxy.rank & (1 << (c_indexes[0] - self.qubits)) and
                    self.proxy.rank & (1 << (c_indexes[1] - self.qubits))
                ):
                    self._algorithm.Diagonal_Multiply_targ(
                        t_index,
                        matrix[MATRIX_INDEXES[7]],
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            elif _c1:
                if self.proxy.rank & (1 << (c_indexes[1] - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[4]]
                    self._algorithm.Controlled_Multiply_ctargs(
                        c_indexes[0],
                        t_index,
                        temp_matrix,
                        self._vector,
                        self._qubits,
                        12,
                        self._sync
                    )
            elif _t:
                temp_matrix = cp.zeros((16,), dtype=self._precision)
                temp_matrix[15] = matrix[63] if self.proxy.rank & (1 << (t_index - self.qubits)) else \
                    matrix[54]

                self._algorithm.Controlled_Multiply_ctargs(
                    c_indexes[1],
                    c_indexes[0],
                    temp_matrix,
                    self._vector,
                    self._qubits,
                    8,
                    self._sync
                )
            else:
                self._algorithm.Controlled_Multiply_more(
                    c_indexes,
                    t_index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_3arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            c_index = self.total_qubits - 1 - gate.carg
            if t_indexes[0] > t_indexes[1]:
                t_indexes[0], t_indexes[1] = t_indexes[1], t_indexes[0]

            _c = c_index >= self.qubits
            _t0 = t_indexes[0] >= self.qubits
            _t1 = t_indexes[1] >= self.qubits

            if _c and _t0:
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    _0 = (self.proxy.rank & (1 << (t_indexes[0] - self.qubits))) >> \
                        (t_indexes[0] - self.qubits)
                    _1 = self.proxy.rank & (1 << (t_indexes[1] - self.qubits)) >> \
                        (t_indexes[1] - self.qubits)

                    if _0 != _1:
                        destination = self.proxy.rank ^ \
                            ((1 << (t_indexes[1] - self.qubits)) + (1 << (t_indexes[0] - self.qubits)))
                        self._data_switcher.all_switch(self._vector, destination)
            elif _c and _t1:
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    destination = self.proxy.rank ^ (1 << (t_indexes[1] - self.qubits))
                    switch_condition = {
                        t_indexes[0]: int(self.proxy.rank < destination)
                    }
                    self._data_switcher.ctargs_switch(
                        self._vector,
                        destination,
                        switch_condition
                    )
            elif _t0:
                _0 = (self.proxy.rank & (1 << (t_indexes[0] - self.qubits))) >> (t_indexes[0] - self.qubits)
                _1 = self.proxy.rank & (1 << (t_indexes[1] - self.qubits)) >> (t_indexes[1] - self.qubits)

                if _0 != _1:
                    destination = self.proxy.rank ^ \
                        ((1 << (t_indexes[1] - self.qubits)) + (1 << (t_indexes[0] - self.qubits)))
                    self._data_switcher.ctargs_switch(
                        self._vector,
                        destination,
                        {c_index: 1}
                    )
            elif _t1:
                destination = self.proxy.rank ^ (1 << (t_indexes[1] - self.qubits))
                switch_condition = {
                    c_index: 1,
                    t_indexes[0]: int(self.proxy.rank < destination)
                }

                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    switch_condition
                )
            elif _c:
                if self.proxy.rank & (1 << (c_index - self.qubits)):
                    self._algorithm.Controlled_Swap_targs(
                        t_indexes,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
            else:
                self._algorithm.Controlled_Swap_tmore(
                    t_indexes,
                    c_index,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate.type() == GATE_ID["Measure"]:
            index = self.total_qubits - 1 - gate.targ
            prob_result = self._algorithm.Device_Prob_Calculator(
                index,
                self._vector,
                self.qubits,
                self.proxy.rank
            )

            total_prob = self._data_switcher.add_prob(prob_result)
            _0 = random.random()
            _1 = _0 > total_prob
            prob = total_prob.get()

            if index >= self.qubits:
                if not _1:
                    alpha = np.float32(1 / np.sqrt(prob)) if self._precision == np.complex64 else \
                        np.float64(1 / np.sqrt(prob))

                    if self.proxy.rank & (1 << (index - self.qubits)):
                        self._vector = cp.zeros_like(self._vector)
                    else:
                        self._algorithm.Float_Multiply(
                            alpha,
                            self._vector,
                            self._qubits,
                            self._sync
                        )
                else:
                    alpha = np.float32(1 / np.sqrt(1 - prob)) if self._precision == np.complex64 else \
                        np.float64(1 / np.sqrt(1 - prob))

                    if self.proxy.rank & (1 << (index - self.qubits)):
                        self._algorithm.Float_Multiply(
                            alpha,
                            self._vector,
                            self._qubits,
                            self._sync
                        )
                    else:
                        self._vector = cp.zeros_like(self._vector)
            else:
                _ = self._algorithm.MeasureGate_Apply(
                    index,
                    self._vector,
                    self._qubits,
                    self._sync,
                    multigpu_prob=prob
                )

            self.circuit.qubits[gate.targ].measured = _1
        elif gate.type() == GATE_ID["Reset"]:
            index = self.total_qubits - 1 - gate.targ
            prob_result = self._algorithm.Device_Prob_Calculator(
                index,
                self._vector,
                self.qubits,
                self.proxy.rank
            )

            total_prob = self._data_switcher.add_prob(prob_result)
            prob = total_prob.get()

            if index >= self.qubits:
                alpha = np.float64(np.sqrt(prob))

                if alpha < 1e-6:
                    destination = self.proxy.rank ^ (1 << (index - self.qubits))
                    if self.proxy.rank & (1 << (index - self.qubits)):
                        self._data_switcher.all_switch(self._vector, destination)
                    else:
                        self._vector = cp.zeros_like(self._vector)
                        self._data_switcher.all_switch(self._vector, destination)
                else:
                    if self.proxy.rank & (1 << (index - self.qubits)):
                        self._vector = cp.zeros_like(self._vector)
                    else:
                        alpha = np.float64(1 / alpha)
                        self._algorithm.Float_Multiply(
                            alpha,
                            self._vector,
                            self._qubits,
                            self._sync
                        )
            else:
                self._algorithm.ResetGate_Apply(
                    index,
                    self._vector,
                    self._qubits,
                    self._sync,
                    multigpu_prob=prob
                )
        elif gate.type() == GATE_ID["Barrier"]:
            # TODO: Not applied in gate.py.
            pass
        elif (
            gate.type() == GATE_ID["Perm"] or
            gate.type() == GATE_ID["ControlPermMulDetail"] or
            gate.type() == GATE_ID["PermShift"] or
            gate.type() == GATE_ID["ControlPermShift"] or
            gate.type() == GATE_ID["PermMul"] or
            gate.type() == GATE_ID["ControlPermMul"] or
            gate.type() == GATE_ID["PermFx"]
        ):
            if gate.targets >= 5:
                pass
            else:
                indexes = gate.pargs
                swaped_pargs = self.perm_operation(indexes)
                self._algorithm.PermGate_Apply(
                    swaped_pargs,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate.type() == GATE_ID["PermT"]:
            mapping = np.array(gate.pargs)
            count, indexes = [], []
            for idx, map in enumerate(mapping):
                if idx not in count:
                    while idx != map:
                        count.append(map)
                        indexes.append([self.total_qubits - 1 - map, self.total_qubits - 1 - idx])
                        map = mapping[map]

            for t_indexes in indexes:
                self.swap_operation(t_indexes)
        elif gate.type() == GATE_ID["PermFxT"]:
            pass
        elif gate.type() == GATE_ID["Unitary"]:
            # TODO: Use np.dot, matrix*vec = 2^n * 2^n x 2^n * 1.
            pass
        elif gate.type() == GATE_ID["ShorInitial"]:
            # TODO: Not applied yet.
            pass
        else:
            raise KeyError("Unsupported Gate in multi-GPU version.")

    def perm_operation(self, indexes):
        dev_num = self.proxy.ndevs
        current_dev = self.proxy.rank
        iter = len(indexes) // dev_num

        ops, perm_indexes = perm_sort(indexes, dev_num)
        for op in ops:
            operation, sender, destination = op
            if operation == "ALL":
                if sender == current_dev:
                    self._data_switcher.all_switch(
                        self._vector,
                        destination
                    )
            elif operation == "IDX":
                if sender // iter == current_dev:
                    ctargs = {}
                    device_idx = sender % iter

                    temp_iter, len_iter = iter, int(np.log2(iter))
                    for c in range(len_iter):
                        temp_iter //= 2
                        if device_idx >= temp_iter:
                            ctargs[len_iter - 1 - c] = 1
                            device_idx -= temp_iter
                        else:
                            ctargs[len_iter - 1 - c] = 0

                    self._data_switcher.ctargs_switch(
                        self._vector,
                        destination // iter,
                        ctargs
                    )
                if destination // iter == current_dev:
                    ctargs = {}
                    device_idx = destination % iter

                    temp_iter, len_iter = iter, int(np.log2(iter))
                    for c in range(len_iter):
                        temp_iter //= 2
                        if device_idx >= temp_iter:
                            ctargs[len_iter - 1 - c] = 1
                            device_idx -= temp_iter
                        else:
                            ctargs[len_iter - 1 - c] = 0

                    self._data_switcher.ctargs_switch(
                        self._vector,
                        sender // iter,
                        ctargs
                    )

        swaped_indexes = perm_indexes[current_dev * iter:current_dev * iter + iter]
        swaped_pargs = np.argsort(swaped_indexes)

        return swaped_pargs

    def swap_operation(self, indexes):
        if indexes[0] >= self.qubits:
            _0 = self.proxy.rank & (1 << (indexes[0] - self.qubits))
            _1 = self.proxy.rank & (1 << (indexes[1] - self.qubits))

            if _0 != _1:
                destination = self.proxy.rank ^ \
                    ((1 << (indexes[0] - self.qubits)) + (1 << (indexes[1] - self.qubits)))
                self._data_switcher.all_switch(self._vector, destination)
        elif indexes[1] >= self.qubits:
            destination = self.proxy.rank ^ (1 << (indexes[1] - self.qubits))
            switch_condition = {indexes[0]: int(self.proxy.rank < destination)}

            self._data_switcher.ctargs_switch(
                self._vector,
                destination,
                switch_condition
            )
        else:
            self._algorithm.Controlled_Swap_targs(
                indexes,
                self._vector,
                self._qubits,
                self._sync
            )
