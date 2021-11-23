#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/28 下午4:50
# @Author  : Kaiqi Li
# @File    : constant_statevecto_simulator

import numpy as np
import cupy as cp

from QuICT.core import *
from QuICT.ops.utils import LinAlgLoader
from QuICT.simulation import BasicGPUSimulator
from QuICT.simulation.utils import GateType, GATE_TYPE_to_ID


class ConstantStateVectorSimulator(BasicGPUSimulator):
    """
    The simulator for qubits' vector state.

    Args:
        circuit (Circuit): The quantum circuit.
        precision [np.complex64, np.complex128]: The precision for the circuit and qubits.
        gpu_device_id (int): The GPU device ID.
        sync (bool): Sync mode or Async mode.
    """
    def __init__(self, circuit: Circuit, precision=np.complex64, gpu_device_id: int = 0, sync: bool = False):
        BasicGPUSimulator.__init__(self, circuit, precision, gpu_device_id)
        self._sync = sync

        # Initial vector state
        self.initial_vector_state()

        # Initial simulator with limit_qubits
        self._algorithm = LinAlgLoader(device="GPU", enable_gate_kernel=True, enable_multigpu_gate_kernel=False)

    def initial_vector_state(self):
        """
        Initial qubits' vector states.
        """
        vector_size = 1 << int(self._qubits)
        # Special Case for no gate circuit
        if len(self._gates) == 0:
            self._vector = np.zeros(vector_size, dtype=self._precision)
            self._vector[0] = self._precision(1)
            return

        # Initial qubit's states
        with cp.cuda.Device(self._device_id):
            self._vector = cp.zeros(vector_size, dtype=self._precision)
            self._vector.put(0, self._precision(1))

    def run(self) -> np.ndarray:
        """
        Get the state vector of circuit
        """
        with cp.cuda.Device(self._device_id):
            for gate in self._gates:
                self.apply_gate(gate)

        return self.vector

    def apply_gate(self, gate):
        """ Depending on the given quantum gate, apply the target algorithm to calculate the state vector.

        Args:
            gate (Gate): the quantum gate in the circuit.
        """
        gate_type = gate.type()
        default_parameters = (self._vector, self._qubits, self._sync)

        # [H, SX, SY, SW, U2, U3, Rx, Ry]
        if gate_type in GATE_TYPE_to_ID[GateType.matrix_1arg]:
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Based_InnerProduct_targ(
                t_index,
                matrix,
                *default_parameters
            )
        # [RZ, Phase]
        elif gate_type in GATE_TYPE_to_ID[GateType.diagonal_1arg]:
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Diagonal_Multiply_targ(
                t_index,
                matrix,
                *default_parameters
            )
        # [X]
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_1arg]:
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.RDiagonal_Swap_targ(
                t_index,
                *default_parameters
            )
        # [Y]
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_1arg]:
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            self._algorithm.RDiagonal_MultiplySwap_targ(
                t_index,
                matrix,
                *default_parameters
            )
        # [Z, U1, T, T_dagger, S, S_dagger]
        elif gate_type in GATE_TYPE_to_ID[GateType.control_1arg]:
            t_index = self._qubits - 1 - gate.targ
            val = gate.compute_matrix[1, 1]
            self._algorithm.Controlled_Multiply_targ(
                t_index,
                val,
                *default_parameters
            )
        # [CRz]
        elif gate_type == GATE_ID["CRz"]:
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Controlled_Multiply_ctargs(
                c_index,
                t_index,
                matrix,
                *default_parameters
            )
        # [Cz, CU1]
        elif gate_type in GATE_TYPE_to_ID[GateType.control_2arg]:
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            val = gate.compute_matrix[3, 3]
            self._algorithm.Controlled_Product_ctargs(
                c_index,
                t_index,
                val,
                *default_parameters
            )
        # [Rzz]
        elif gate_type in GATE_TYPE_to_ID[GateType.diagonal_2arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Diagonal_Multiply_targs(
                t_indexes,
                matrix,
                *default_parameters
            )
        # [CX, CY]
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_2arg]:
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Controlled_MultiplySwap_ctargs(
                c_index,
                t_index,
                matrix,
                *default_parameters
            )
        # [CH, CU3]
        elif gate_type in GATE_TYPE_to_ID[GateType.matrix_2arg]:
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Controlled_InnerProduct_ctargs(
                c_index,
                t_index,
                matrix,
                *default_parameters
            )
        # [FSim]
        elif gate_type in GATE_TYPE_to_ID[GateType.complexMIP_2arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Completed_MxIP_targs(
                t_indexes,
                matrix,
                *default_parameters
            )
        # [Rxx, Ryy]
        elif gate_type in GATE_TYPE_to_ID[GateType.complexIPIP_2arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Completed_IPxIP_targs(
                t_indexes,
                matrix,
                *default_parameters
            )
        # [Swap]
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_2arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            self._algorithm.Controlled_Swap_targs(
                t_indexes,
                *default_parameters
            )
        # [ID]
        elif gate.type() == GATE_ID["ID"]:
            pass
        # [CCX]
        elif gate_type in GATE_TYPE_to_ID[GateType.reverse_3arg]:
            c_indexes = [self._qubits - 1 - carg for carg in gate.cargs]
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.Controlled_Swap_more(
                c_indexes,
                t_index,
                *default_parameters
            )
        # [CCRz]
        elif gate_type in GATE_TYPE_to_ID[GateType.control_3arg]:
            c_indexes = [self._qubits - 1 - carg for carg in gate.cargs]
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Controlled_Multiply_more(
                c_indexes,
                t_index,
                matrix,
                *default_parameters
            )
        # [CSwap]
        elif gate_type in GATE_TYPE_to_ID[GateType.swap_3arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            c_index = self._qubits - 1 - gate.carg
            self._algorithm.Controlled_Swap_tmore(
                t_indexes,
                c_index,
                *default_parameters
            )
        # [Measure]
        elif gate_type == GATE_ID["Measure"]:
            index = self._qubits - 1 - gate.targ
            result = self._algorithm.MeasureGate_Apply(
                index,
                *default_parameters
            )
            self.circuit.qubits[gate.targ].measured = result
        # [Reset]
        elif gate_type == GATE_ID["Reset"]:
            index = self._qubits - 1 - gate.targ
            self._algorithm.ResetGate_Apply(
                index,
                *default_parameters
            )
        # [Barrier]
        elif gate_type == GATE_ID["Barrier"]:
            # TODO: Not applied in gate.py.
            pass
        # TODO: only working with the Perm gate
        elif (
            gate_type == GATE_ID["Perm"] or
            gate_type == GATE_ID["ControlPermMulDetail"] or
            gate_type == GATE_ID["PermShift"] or
            gate_type == GATE_ID["ControlPermShift"] or
            gate_type == GATE_ID["PermMul"] or
            gate_type == GATE_ID["ControlPermMul"] or
            gate_type == GATE_ID["PermFx"]
        ):
            if gate.targets >= 12:
                pass
            else:
                self._algorithm.PermGate_Apply(
                    gate.pargs,
                    *default_parameters
                )
        # extra gates. i.e. [Unitary, Customied]
        else:
            aux = cp.zeros_like(self._vector)
            matrix = self.get_gate_matrix(gate)
            self._algorithm.matrix_dot_vector(
                matrix,
                gate.controls + gate.targets,
                self._vector,
                self._qubits,
                gate.affectArgs,
                aux,
                self._sync
            )
            self.vector = aux
