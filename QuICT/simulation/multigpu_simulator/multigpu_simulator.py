#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/30 下午5:50
# @Author  : Kaiqi Li
# @File    : proxy_simulator

import numpy as np
import cupy as cp

from QuICT.core import *
from QuICT.simulation import BasicGPUSimulator
from QuICT.utility import Proxy
from QuICT.ops.utils import LinAlgLoader, perm_sort
from QuICT.simulation.utils import GateType, GATE_TYPE_to_ID, MATRIX_INDEXES
from QuICT.simulation.multigpu_simulator.data_switch import DataSwitcher


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
        assert(proxy.dev_id == gpu_device_id)

        # Initial simulator with qubits
        BasicGPUSimulator.__init__(self, circuit, precision, gpu_device_id)

        # Get qubits and limitation
        self.total_qubits = int(circuit.circuit_width())
        self.qubits = int(self.total_qubits - np.log2(self.proxy.ndevs))

        # Initial GateMatrix
        self._gate_matrix_prepare()

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
            if self.proxy.dev_id == 0:
                self._vector[0] = self._precision(1)
            return

        # Initial qubit's states
        with cp.cuda.Device(self._device_id):
            self._vector = cp.zeros(vector_size, dtype=self._precision)
            if self.proxy.dev_id == 0:
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
        """ Depending on the given quantum gate, apply the target algorithm to calculate the state vector.
        If the target indexes or the control index of the quantum gate exceed the limit of device qubits, applied
        the special algorithm for this situation.

        e.g.
        For the quantum gate with the diagonal matirx: [[a, 0],
                                                        [0, b]],
        and its target index is exceed the limit.
        For the device contains the state which the target index is 0, it will apply V * a;
        the device contains the state which the target index is 1, it will apply V * b.

        Args:
            gate (Gate): the quantum gate in the circuit.
        """
        gate_type = gate.type()
        default_parameters = (self._vector, self._qubits, self._sync)

        # [H, SX, SY, SW, U2, U3, Rx, Ry]
        if gate_type in GATE_TYPE_to_ID[GateType.matrix_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)

            self._based_matrix_operation(t_index, matrix)
        # [RZ, Phase]
        elif gate_type in GATE_TYPE_to_ID[GateType.diagonal_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate) if t_index < self.qubits else gate.compute_matrix

            self._diagonal_matrix_operation(t_index, matrix)
        # [X]
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_1arg]:
            t_index = self.total_qubits - 1 - gate.targ

            self._swap_matrix_operation(t_index)
        # [Y]
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate) if t_index < self.qubits else gate.compute_matrix

            self._reverse_matrix_operation(t_index, matrix)
        # [Z, U1, T, T_dagger, S, S_dagger]
        elif gate_type in GATE_TYPE_to_ID[GateType.control_1arg]:
            t_index = self.total_qubits - 1 - gate.targ
            val = gate.compute_matrix[1, 1]

            self._control_matrix_operation(t_index, val)
        # [CRz]
        elif gate_type == GATE_ID["CRz"]:
            t_index = self.total_qubits - 1 - gate.targ
            c_index = self.total_qubits - 1 - gate.carg

            # target index and control index both exceed the device limit.
            if t_index >= self.qubits and c_index >= self.qubits:
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    value = gate.compute_matrix[3, 3] if self.proxy.dev_id & (1 << (t_index - self.qubits)) else \
                        gate.compute_matrix[2, 2]
                    self._algorithm.Simple_Multiply(
                        value,
                        *default_parameters
                    )
            elif c_index >= self.qubits:    # control index exceed the device limt.
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    matrix = self.get_gate_matrix(gate)
                    temp_matrix = matrix[MATRIX_INDEXES[0]]
                    self._algorithm.Diagonal_Multiply_targ(
                        t_index,
                        temp_matrix,
                        *default_parameters
                    )
            elif t_index >= self.qubits:    # target index exceed the device limit
                value = gate.compute_matrix[3, 3] if self.proxy.dev_id & (1 << (t_index - self.qubits)) else \
                    gate.compute_matrix[2, 2]
                self._algorithm.Controlled_Multiply_targ(
                    c_index,
                    value,
                    *default_parameters
                )
            else:
                matrix = self.get_gate_matrix(gate)
                self._algorithm.Controlled_Multiply_ctargs(
                    c_index,
                    t_index,
                    matrix,
                    *default_parameters
                )
        # [CZ, CU1]
        elif gate_type in GATE_TYPE_to_ID[GateType.control_2arg]:
            t_index = self.total_qubits - 1 - gate.targ
            c_index = self.total_qubits - 1 - gate.carg
            value = gate.compute_matrix[3, 3]

            # target index and control index both exceed the device limit.
            if t_index >= self.qubits and c_index >= self.qubits:
                if (
                    self.proxy.dev_id & (1 << (c_index - self.qubits)) and
                    self.proxy.dev_id & (1 << (t_index - self.qubits))
                ):
                    self._algorithm.Simple_Multiply(
                        value,
                        *default_parameters
                    )
            elif c_index >= self.qubits:    # control index exceed the device limt.
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    self._algorithm.Controlled_Multiply_targ(
                        t_index,
                        value,
                        *default_parameters
                    )
            elif t_index >= self.qubits:    # target index exceed the device limt.
                if self.proxy.dev_id & (1 << (t_index - self.qubits)):
                    self._algorithm.Controlled_Multiply_targ(
                        c_index,
                        value,
                        *default_parameters
                    )
            else:
                self._algorithm.Controlled_Product_ctargs(
                    c_index,
                    t_index,
                    value,
                    *default_parameters
                )
        # [Rzz]
        elif gate_type in GATE_TYPE_to_ID[GateType.diagonal_2arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            t_indexes.sort()
            matrix = self.get_gate_matrix(gate) if t_indexes[0] < self.qubits else gate.compute_matrix

            self._diagonal_2args_matrix_operation(
                t_indexes,
                matrix
            )
        # [CX, CY]
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_2arg]:
            t_index = self.total_qubits - 1 - gate.targ
            c_index = self.total_qubits - 1 - gate.carg

            # target index and control index both exceed the device limit.
            if t_index >= self.qubits and c_index >= self.qubits:
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    index = (2, 3) if self.proxy.dev_id & (1 << (t_index - self.qubits)) else \
                        (3, 2)
                    self.Simple_Multiply(
                        gate.compute_matrix[index],
                        *default_parameters
                    )

                    destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
                    self._data_switcher.all_switch(self._vector, destination)
            elif t_index >= self.qubits:    # # target index exceed the device limit.
                value = gate.compute_matrix[2, 3] if self.proxy.dev_id & (1 << (t_index - self.qubits)) else \
                    gate.compute_matrix[3, 2]

                self._algorithm.Controlled_Multiply_targ(
                    c_index,
                    value,
                    *default_parameters
                )

                destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    {c_index: 1}
                )
            elif c_index >= self.qubits:    # control index both exceed the device limit.
                matrix = self.get_gate_matrix(gate)
                temp_matrix = cp.zeros((4,), dtype=self._precision)
                temp_matrix[1], temp_matrix[2] = matrix[11], matrix[14]
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    self._algorithm.RDiagonal_MultiplySwap_targ(
                        t_index,
                        temp_matrix,
                        *default_parameters
                    )
            else:
                matrix = self.get_gate_matrix(gate)
                self._algorithm.Controlled_MultiplySwap_ctargs(
                    c_index,
                    t_index,
                    matrix,
                    *default_parameters
                )
        # [CH, CU3]
        elif gate_type in GATE_TYPE_to_ID[GateType.matrix_2arg]:
            t_index = self.total_qubits - 1 - gate.targ
            c_index = self.total_qubits - 1 - gate.carg
            matrix = self.get_gate_matrix(gate)

            # target index and control index both exceed the device limit.
            if t_index >= self.qubits and c_index >= self.qubits:
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
                    self._data_switcher.half_switch(
                        self._vector,
                        destination
                    )

                    self._algorithm.Based_InnerProduct_targ(
                        self.qubits - 1,
                        matrix[MATRIX_INDEXES[0]],
                        *default_parameters
                    )

                    self._data_switcher.half_switch(
                        self._vector,
                        destination
                    )
            elif c_index >= self.qubits:    # control index both exceed the device limit.
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[0]]
                    self._algorithm.Based_InnerProduct_targ(
                        t_index,
                        temp_matrix,
                        *default_parameters
                    )
            elif t_index >= self.qubits:    # target index exceed the device limit.
                destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
                switch_condition = {c_index: int(self.proxy.dev_id < destination)}

                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    switch_condition
                )

                if self.proxy.dev_id & (1 << (t_index - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[0]]
                    self._algorithm.Based_InnerProduct_targ(
                        c_index,
                        temp_matrix,
                        *default_parameters
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
                    *default_parameters
                )
        # [FSim]
        elif gate_type in GATE_TYPE_to_ID[GateType.complexMIP_2arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            t_indexes.sort()
            matrix = self.get_gate_matrix(gate)

            # target indexes exceed the device limit.
            if t_indexes[0] >= self.qubits:
                _0 = self.proxy.dev_id & (1 << (t_indexes[0] - self.qubits))
                _1 = self.proxy.dev_id & (1 << (t_indexes[1] - self.qubits))

                if _0 and _1:
                    self._algorithm.Simple_Multiply(
                        gate.compute_matrix[3, 3],
                        *default_parameters
                    )
                elif _0 or _1:
                    destination = self.proxy.dev_id ^ \
                        ((1 << (t_indexes[0] - self.qubits)) + (1 << (t_indexes[1] - self.qubits)))
                    self._data_switcher.half_switch(
                        self._vector,
                        destination
                    )
                    self._algorithm.Based_InnerProduct_targ(
                        self.qubits - 1,
                        matrix[MATRIX_INDEXES[1]],
                        *default_parameters
                    )
                    self._data_switcher.half_switch(
                        self._vector,
                        destination
                    )
                else:
                    pass
            elif t_indexes[1] >= self.qubits:   # larger target index exceed the device limit.
                destination = self.proxy.dev_id ^ (1 << (t_indexes[1] - self.qubits))
                self._data_switcher.ctargs_switch(
                    self.vector,
                    destination,
                    {t_indexes[0]: 1}
                )

                if self.proxy.dev_id & (1 << (t_indexes[1] - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[2]]
                    self._algorithm.Based_InnerProduct_targ(
                        t_indexes[0],
                        temp_matrix,
                        *default_parameters
                    )
                else:
                    self._algorithm.Controlled_Multiply_targ(
                        t_indexes[0],
                        gate.compute_matrix[3, 3],
                        *default_parameters
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
                    *default_parameters
                )
        # [Rxx, Ryy]
        elif gate_type in GATE_TYPE_to_ID[GateType.complexIPIP_2arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            t_indexes.sort()
            matrix = self.get_gate_matrix(gate)

            # target indexes both exceed the device limit.
            if t_indexes[0] >= self.qubits:
                destination = self.proxy.dev_id ^ \
                    ((1 << (t_indexes[0] - self.qubits)) + (1 << (t_indexes[1] - self.qubits)))
                self._data_switcher.half_switch(
                    self._vector,
                    destination
                )

                if (
                    self.proxy.dev_id & (1 << (t_indexes[0] - self.qubits)) ==
                    self.proxy.dev_id & (1 << (t_indexes[1] - self.qubits))
                ):
                    temp_matrix = matrix[MATRIX_INDEXES[3]]
                else:
                    temp_matrix = matrix[MATRIX_INDEXES[1]]

                self._algorithm.Based_InnerProduct_targ(
                    self.qubits - 1,
                    temp_matrix,
                    *default_parameters
                )

                self._data_switcher.half_switch(
                    self._vector,
                    destination
                )
            elif t_indexes[1] >= self.qubits:   # larger target index exceed the device limit.
                destination = self.proxy.dev_id ^ (1 << (t_indexes[1] - self.qubits))
                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    {t_indexes[0]: 1}
                )

                if self.proxy.dev_id & (1 << (t_indexes[1] - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[2]]
                else:
                    temp_matrix = matrix[MATRIX_INDEXES[3]]

                self._algorithm.Based_InnerProduct_targ(
                    t_indexes[0],
                    temp_matrix,
                    *default_parameters
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
                    *default_parameters
                )
        # [Swap]
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_2arg]:
            indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            indexes.sort()

            if indexes[0] >= self.qubits:   # index 0 exceed the limit
                _0 = self.proxy.dev_id & (1 << (indexes[0] - self.qubits))
                _1 = self.proxy.dev_id & (1 << (indexes[1] - self.qubits))

                if _0 != _1:
                    destination = self.proxy.dev_id ^ \
                        ((1 << (indexes[0] - self.qubits)) + (1 << (indexes[1] - self.qubits)))
                    self._data_switcher.all_switch(self._vector, destination)
            elif indexes[1] >= self.qubits:     # index 1 exceed the limit
                destination = self.proxy.dev_id ^ (1 << (indexes[1] - self.qubits))
                switch_condition = {indexes[0]: int(self.proxy.dev_id < destination)}

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
        # [ID]
        elif gate_type == GATE_ID["ID"]:
            pass
        # [CCX]
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_3arg]:
            c_indexes = [self.total_qubits - 1 - carg for carg in gate.cargs]
            c_indexes.sort()
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)

            _t = t_index >= self.qubits         # target index exceed the limit?
            _c0 = c_indexes[0] >= self.qubits   # control index exceed the limit?
            _c1 = c_indexes[1] >= self.qubits   # larger control index exceed the limit?

            if _t and _c0:
                if (
                    self.proxy.dev_id & (1 << (c_indexes[0] - self.qubits)) and
                    self.proxy.dev_id & (1 << (c_indexes[1] - self.qubits))
                ):
                    destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
                    self._data_switcher.all_switch(self._vector, destination)
            elif _t and _c1:
                if self.proxy.dev_id & (1 << (c_indexes[1] - self.qubits)):
                    destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
                    switch_condition = {c_indexes[0]: 1}
                    self._data_switcher.ctargs_switch(
                        self._vector,
                        destination,
                        switch_condition
                    )
            elif _c0:
                if (
                    self.proxy.dev_id & (1 << (c_indexes[0] - self.qubits)) and
                    self.proxy.dev_id & (1 << (c_indexes[1] - self.qubits))
                ):
                    self._algorithm.RDiagonal_Swap_targ(
                        t_index,
                        *default_parameters
                    )
            elif _c1:
                if self.proxy.dev_id & (1 << (c_indexes[1] - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[4]]
                    self._algorithm.Controlled_MultiplySwap_ctargs(
                        c_indexes[0],
                        t_index,
                        temp_matrix,
                        *default_parameters
                    )
            elif _t:
                destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
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
                    *default_parameters
                )
        # [CCRz]
        elif gate_type in GATE_TYPE_to_ID[GateType.control_3arg]:
            c_indexes = [self.total_qubits - 1 - carg for carg in gate.cargs]
            c_indexes.sort()
            t_index = self.total_qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)

            _t = t_index >= self.qubits         # target index exceed the limit?
            _c0 = c_indexes[0] >= self.qubits   # control index exceed the limit?
            _c1 = c_indexes[1] >= self.qubits   # larger control index exceed the limit?

            if _t and _c0:
                if (
                    self.proxy.dev_id & (1 << (c_indexes[0] - self.qubits)) and
                    self.proxy.dev_id & (1 << (c_indexes[1] - self.qubits))
                ):
                    index = (7, 7) if self.proxy.dev_id & (1 << (t_index - self.qubits)) else \
                        (6, 6)

                    self._algorithm.Simple_Multiply(
                        gate.compute_matrix[index],
                        *default_parameters
                    )
            elif _t and _c1:
                if self.proxy.dev_id & (1 << (c_indexes[1] - self.qubits)):
                    if self.proxy.dev_id & (1 << (t_index - self.qubits)):
                        value = gate.compute_matrix[7, 7]
                    else:
                        value = gate.compute_matrix[6, 6]
                    self._algorithm.Controlled_Multiply_targ(
                        c_indexes[0],
                        value,
                        *default_parameters
                    )
            elif _c0:
                if (
                    self.proxy.dev_id & (1 << (c_indexes[0] - self.qubits)) and
                    self.proxy.dev_id & (1 << (c_indexes[1] - self.qubits))
                ):
                    self._algorithm.Diagonal_Multiply_targ(
                        t_index,
                        matrix[MATRIX_INDEXES[7]],
                        *default_parameters
                    )
            elif _c1:
                if self.proxy.dev_id & (1 << (c_indexes[1] - self.qubits)):
                    temp_matrix = matrix[MATRIX_INDEXES[4]]
                    self._algorithm.Controlled_Multiply_ctargs(
                        c_indexes[0],
                        t_index,
                        temp_matrix,
                        *default_parameters
                    )
            elif _t:
                index = (7, 7) if self.proxy.dev_id & (1 << (t_index - self.qubits)) else \
                    (6, 6)

                self._algorithm.Controlled_Product_ctargs(
                    c_indexes[1],
                    c_indexes[0],
                    gate.compute_matrix[index],
                    *default_parameters
                )
            else:
                self._algorithm.Controlled_Multiply_more(
                    c_indexes,
                    t_index,
                    matrix,
                    *default_parameters
                )
        # [CSwap]
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_3arg]:
            t_indexes = [self.total_qubits - 1 - targ for targ in gate.targs]
            t_indexes.sort()
            c_index = self.total_qubits - 1 - gate.carg

            _c = c_index >= self.qubits         # control index exceed the limit?
            _t0 = t_indexes[0] >= self.qubits   # target index exceed the limit?
            _t1 = t_indexes[1] >= self.qubits   # larger target index exceed the limit?

            if _c and _t0:
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    _0 = (self.proxy.dev_id & (1 << (t_indexes[0] - self.qubits))) >> \
                        (t_indexes[0] - self.qubits)
                    _1 = self.proxy.dev_id & (1 << (t_indexes[1] - self.qubits)) >> \
                        (t_indexes[1] - self.qubits)

                    if _0 != _1:
                        destination = self.proxy.dev_id ^ \
                            ((1 << (t_indexes[1] - self.qubits)) + (1 << (t_indexes[0] - self.qubits)))
                        self._data_switcher.all_switch(self._vector, destination)
            elif _c and _t1:
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    destination = self.proxy.dev_id ^ (1 << (t_indexes[1] - self.qubits))
                    switch_condition = {
                        t_indexes[0]: int(self.proxy.dev_id < destination)
                    }
                    self._data_switcher.ctargs_switch(
                        self._vector,
                        destination,
                        switch_condition
                    )
            elif _t0:
                _0 = (self.proxy.dev_id & (1 << (t_indexes[0] - self.qubits))) >> (t_indexes[0] - self.qubits)
                _1 = self.proxy.dev_id & (1 << (t_indexes[1] - self.qubits)) >> (t_indexes[1] - self.qubits)

                if _0 != _1:
                    destination = self.proxy.dev_id ^ \
                        ((1 << (t_indexes[1] - self.qubits)) + (1 << (t_indexes[0] - self.qubits)))
                    self._data_switcher.ctargs_switch(
                        self._vector,
                        destination,
                        {c_index: 1}
                    )
            elif _t1:
                destination = self.proxy.dev_id ^ (1 << (t_indexes[1] - self.qubits))
                switch_condition = {
                    c_index: 1,
                    t_indexes[0]: int(self.proxy.dev_id < destination)
                }

                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    switch_condition
                )
            elif _c:
                if self.proxy.dev_id & (1 << (c_index - self.qubits)):
                    self._algorithm.Controlled_Swap_targs(
                        t_indexes,
                        *default_parameters
                    )
            else:
                self._algorithm.Controlled_Swap_tmore(
                    t_indexes,
                    c_index,
                    *default_parameters
                )
        # [Measure]
        elif gate_type == GATE_ID["Measure"]:
            index = self.total_qubits - 1 - gate.targ

            self.circuit.qubits[gate.targ].measured = self._measure_operation(index)
        # [Reset]
        elif gate_type == GATE_ID["Reset"]:
            index = self.total_qubits - 1 - gate.targ

            self._reset_operation(index)
        # [Barrier]
        elif gate_type == GATE_ID["Barrier"]:
            # TODO: Not applied in gate.py.
            pass
        # [Perm]
        elif (
            gate_type == GATE_ID["Perm"] or
            gate_type == GATE_ID["ControlPermMulDetail"] or
            gate_type == GATE_ID["PermShift"] or
            gate_type == GATE_ID["ControlPermShift"] or
            gate_type == GATE_ID["PermMul"] or
            gate_type == GATE_ID["ControlPermMul"] or
            gate_type == GATE_ID["PermFx"]
        ):
            if gate.targets >= 5:
                pass
            else:
                indexes = gate.pargs
                swaped_pargs = self._perm_operation(indexes)
                self._algorithm.PermGate_Apply(
                    swaped_pargs,
                    *default_parameters
                )
        # [Unitary]
        elif gate_type == GATE_ID["Unitary"]:
            # TODO: Currently only support the 1-, 2-qubits unitary quantum gates
            qubit_idxes = gate.cargs + gate.targs
            if len(qubit_idxes) == 1:   # 1-qubit unitary gate
                if gate.is_diagonal():    # diagonal gate
                    t_index = self.total_qubits - 1 - qubit_idxes[0]
                    matrix = self.get_gate_matrix(gate) if t_index < self.qubits else gate.compute_matrix

                    self._diagonal_matrix_operation(t_index, matrix)
                else:   # non-diagonal gate
                    t_index = self.total_qubits - 1 - qubit_idxes[0]
                    matrix = self.get_gate_matrix(gate)

                    self._based_matrix_operation(t_index, matrix)
            elif len(qubit_idxes) == 2:     # 2-qubits unitary gate
                indexes = [self.total_qubits - 1 - index for index in qubit_idxes]
                indexes.sort()
                matrix = self.get_gate_matrix(gate) if indexes[0] < self.qubits else gate.compute_matrix
                if gate.is_diagonal():        # diagonal gate
                    self._diagonal_2args_matrix_operation(
                        indexes,
                        matrix
                    )
                else:   # non-diagonal gate
                    self._based_2args_matrix_operation(
                        indexes,
                        matrix
                    )
            else:   # common unitary gate supported, but run slowly
                raise ValueError(
                    "do not support the unitary gate with more than 2 qubits, please use gate decomposition first."
                )
        # [ShorInitial]
        elif gate_type == GATE_ID["ShorInitial"]:
            # TODO: Not applied yet.
            pass
        # unsupported quantum gates
        else:
            raise KeyError("Unsupported Gate in multi-GPU version: {gate_type}.")

    def _based_matrix_operation(self, index, matrix):
        """ The algorithm for 1-qubit non-diagonal quantum gate.

        Args:
            index (int): The target qubit of the applied quantum gate.
            matrix (cp.array): The compute matrix of the applied quantum gate.
        """
        default_parameters = (self._vector, self._qubits, self._sync)
        is_exceed = (index >= self.qubits)

        if is_exceed:  # target index exceed the device limit; switch data with dest
            destination = self.proxy.dev_id ^ (1 << (index - self.qubits))
            self._data_switcher.half_switch(self._vector, destination)
            index = self.qubits - 1

        self._algorithm.Based_InnerProduct_targ(
            index,
            matrix,
            *default_parameters
        )

        if is_exceed:   # Switch data back from the destination
            self._data_switcher.half_switch(self._vector, destination)

    def _based_2args_matrix_operation(self, indexes, matrix):
        """ The algorithm for 2-qubits non-diagonal quantum gate.

        Args:
            index (int): The target/control qubits of the applied quantum gate.
            matrix (cp.array): The compute matrix of the applied quantum gate.
        """
        default_parameters = (self._vector, self._qubits, self._sync)

        if indexes[0] >= self.qubits:       # all qubits exceed the device limit
            curr_dev_id = self.proxy.dev_id
            # Get all destination for the current device.
            destination = np.array([
                curr_dev_id,
                curr_dev_id ^ (1 << (indexes[0] - self.qubits)),    # destination with different indexes[0]
                curr_dev_id ^ (1 << (indexes[1] - self.qubits)),    # destination with different indexes[1]
                curr_dev_id ^ (1 << (indexes[0] - self.qubits) + 1 << (indexes[1] - self.qubits))
            ])
            destination = np.sort(destination)

            # Switch data with other destinations
            self._data_switcher.quarter_switch(self._vector, destination)

            self._algorithm.Based_InnerProduct_targs(
                [self.qubits - 2, self.qubits - 1],
                matrix,
                *default_parameters
            )
        elif indexes[1] >= self.qubits:     # if one qubit exceed the device limit
            destination = self.proxy.dev_id ^ (1 << (indexes[1] - self.qubits))
            if indexes[0] != self.qubits - 1:   # swap data by the index not equal to indexes[0]
                self._data_switcher.half_switch(self._vector, destination)
                indexes[1] = self.qubits - 1
            else:
                _0_1 = 1 if not self.proxy.dev_id & (1 << (indexes[1] - self.qubits)) else 0
                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    {0: _0_1}
                )
                indexes[1] = 0

            self._algorithm.Based_InnerProduct_targs(
                indexes,
                matrix,
                *default_parameters
            )

            # Swap data back to the destination
            if indexes[0] != self.qubits - 1:
                self._data_switcher.half_switch(self._vector, destination)
            else:
                self._data_switcher.ctargs_switch(
                    self._vector,
                    destination,
                    {0: _0_1}
                )
        else:   # no qubits exceed the device limit.
            self._algorithm.Based_InnerProduct_targs(
                indexes,
                matrix,
                *default_parameters
            )

    def _diagonal_matrix_operation(self, t_index, matrix):
        """ The algorithm for 1-qubit diagonal quantum gate.

        Args:
            index (int): The target qubit of the applied quantum gate.
            matrix (cp.array): The compute matrix of the applied quantum gate.
        """
        default_parameters = (self._vector, self._qubits, self._sync)

        if t_index >= self.qubits:  # target index exceed the device limit
            index = self.proxy.dev_id & (1 << (t_index - self.qubits))
            value = matrix[index, index]

            self._algorithm.Simple_Multiply(
                value,
                *default_parameters
            )
        else:
            self._algorithm.Diagonal_Multiply_targ(
                t_index,
                matrix,
                *default_parameters
            )

    def _diagonal_2args_matrix_operation(self, indexes, matrix):
        """ The algorithm for 2-qubits diagonal quantum gate.

        Args:
            indexes (int): The target/control qubits of the applied quantum gate.
            matrix (cp.array): The compute matrix of the applied quantum gate.
        """
        default_parameters = (self._vector, self._qubits, self._sync)

        if indexes[0] >= self.qubits:       # All qubits exceed the device limit.
            _0 = self.proxy.dev_id & (1 << (indexes[0] - self.qubits))
            _1 = self.proxy.dev_id & (1 << (indexes[1] - self.qubits))

            index = 0
            if _0:
                index += 1
            if _1:
                index += 2

            self._algorithm.Simple_Multiply(
                matrix[index, index],
                *default_parameters
            )
        elif indexes[1] >= self.qubits:     # one qubit exceed the device limit
            temp_matrix = cp.zeros((4,), dtype=self._precision)
            if self.proxy.dev_id & (1 << (indexes[1] - self.qubits)):
                temp_matrix[0], temp_matrix[3] = matrix[10], matrix[15]
            else:
                temp_matrix[0], temp_matrix[3] = matrix[0], matrix[5]

            self._algorithm.Diagonal_Multiply_targ(
                indexes[0],
                temp_matrix,
                *default_parameters
            )
        else:       # no qubits exceed the device limit
            self._algorithm.Diagonal_Multiply_targs(
                indexes,
                matrix,
                *default_parameters
            )

    def _swap_matrix_operation(self, t_index):
        """ The algorithm for the 1-qubit swap quantum gate

        Args:
            t_index (int): The target qubit of the applied quantum gate.
        """
        default_parameters = (self._vector, self._qubits, self._sync)

        if t_index >= self.qubits:  # Swap the whole data if target idx is exceed.
            destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
            self._data_switcher.all_switch(self.vector, destination)
        else:
            self._algorithm.RDiagonal_Swap_targ(
                t_index,
                *default_parameters
            )

    def _reverse_matrix_operation(self, t_index, matrix):
        """ The algorithm of the 1-qubit reverse quantum gate.

        Args:
            index (int): The target qubit of the applied quantum gate.
            matrix (cp.array): The compute matrix of the applied quantum gate.
        """
        default_parameters = (self._vector, self._qubits, self._sync)

        if t_index >= self.qubits:  # target index exceed the device limit
            destination = self.proxy.dev_id ^ (1 << (t_index - self.qubits))
            index = (0, 1) if self.proxy.dev_id & (1 << (t_index - self.qubits)) else \
                (1, 0)
            value = matrix[index]

            self._algorithm.Simple_Multiply(
                value,
                *default_parameters
            )

            self._data_switcher.all_switch(self._vector, destination)
        else:
            self._algorithm.RDiagonal_MultiplySwap_targ(
                t_index,
                matrix,
                *default_parameters
            )

    def _control_matrix_operation(self, t_index, value):
        """ The algorithm of the 1-qubit quantum gate with controlled matrix.
        e.g. the quantum gate with matrix  [[1, 0],
                                            [0, a]]

        Args:
            index (int): The target qubit of the applied quantum gate.
            matrix (cp.array): The compute matrix of the applied quantum gate.
        """
        default_parameters = (self._vector, self._qubits, self._sync)

        if t_index >= self.qubits:  # target index exceed the device limit.
            if self.proxy.dev_id & (1 << (t_index - self.qubits)):
                self._algorithm.Simple_Multiply(
                    value,
                    *default_parameters
                )
        else:
            self._algorithm.Controlled_Multiply_targ(
                t_index,
                value,
                *default_parameters
            )

    def _measure_operation(self, index):
        """ The algorithm for the Measure gate.

        Args:
            index (int): The target qubit of the applied quantum gate.

        Returns:
            [bool]: state 0 or state 1
        """
        default_parameters = (self._vector, self._qubits, self._sync)

        # Calculate the device's probability.
        prob_result = self._algorithm.Device_Prob_Calculator(
            index,
            self._vector,
            self.qubits,
            self.proxy.dev_id
        )

        # Combined the probability for all device
        total_prob = self._data_switcher.add_prob(prob_result)
        _0 = random.random()        # random mistake
        _1 = _0 > total_prob        # result in state 0 or state 1
        prob = total_prob.get()

        if index >= self.qubits:    # target index exceed the limit
            if _1:  # result in state 1
                alpha = np.float32(1 / np.sqrt(1 - prob)) if self._precision == np.complex64 else \
                    np.float64(1 / np.sqrt(1 - prob))

                if self.proxy.dev_id & (1 << (index - self.qubits)):
                    self._algorithm.Float_Multiply(
                        alpha,
                        *default_parameters
                    )
                else:
                    self._vector = cp.zeros_like(self._vector)
            else:       # result in state 0
                alpha = np.float32(1 / np.sqrt(prob)) if self._precision == np.complex64 else \
                    np.float64(1 / np.sqrt(prob))

                if self.proxy.dev_id & (1 << (index - self.qubits)):
                    self._vector = cp.zeros_like(self._vector)
                else:
                    self._algorithm.Float_Multiply(
                        alpha,
                        *default_parameters
                    )
        else:
            self._algorithm.MeasureGate_Apply(
                index,
                *default_parameters,
                multigpu_prob=prob
            )

        return _1

    def _reset_operation(self, index):
        """ The algorithm for the Reset gate.

        Args:
            index (int): The target qubit of the applied quantum gate.
        """
        default_parameters = (self._vector, self._qubits, self._sync)

        # Calculate the device's probability
        prob_result = self._algorithm.Device_Prob_Calculator(
            index,
            self._vector,
            self.qubits,
            self.proxy.dev_id
        )

        total_prob = self._data_switcher.add_prob(prob_result)
        prob = total_prob.get()

        if index >= self.qubits:
            alpha = np.float64(np.sqrt(prob))

            if alpha < 1e-6:
                destination = self.proxy.dev_id ^ (1 << (index - self.qubits))
                if not (self.proxy.dev_id & (1 << (index - self.qubits))):
                    self._vector = cp.zeros_like(self._vector)
                    self._data_switcher.all_switch(self._vector, destination)
            else:
                if self.proxy.dev_id & (1 << (index - self.qubits)):
                    self._vector = cp.zeros_like(self._vector)
                else:
                    alpha = np.float64(1 / alpha)
                    self._algorithm.Float_Multiply(
                        alpha,
                        *default_parameters
                    )
        else:
            self._algorithm.ResetGate_Apply(
                index,
                *default_parameters,
                multigpu_prob=prob
            )

    def _perm_operation(self, indexes):
        """ Temporary algorithm for the PermGate.

        Args:
            indexes ([int]): the parameter args in the PermGate

        Returns:
            [int]: the state indexes after permutation.
        """
        dev_num = self.proxy.ndevs
        current_dev = self.proxy.dev_id
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
