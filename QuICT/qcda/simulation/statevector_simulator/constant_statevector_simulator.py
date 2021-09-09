#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/28 下午4:50
# @Author  : Kaiqi Li
# @File    : constant_statevecto_simulator

import numpy as np
import cupy as cp

from QuICT.ops.utils import LinAlgLoader
from QuICT.qcda.simulation import BasicSimulator
from QuICT.core import *


class ConstantStateVectorSimulator(BasicSimulator):
    """
    The simulator for qubits' vector state.

    Args:
        circuit (Circuit): The quantum circuit.
        precision [np.complex64, np.complex128]: The precision for the circuit and qubits.
        gpu_device_id (int): The GPU device ID.
        sync (bool): Sync mode or Async mode.
    """
    def __init__(self, circuit: Circuit, precision = np.complex64, gpu_device_id: int = 0, sync: bool = False):
        BasicSimulator.__init__(self, circuit, precision, gpu_device_id)
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
            self._vector = cp.empty(vector_size, dtype=self._precision)
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
        if (
            gate.type() == GATE_ID["H"] or 
            gate.type() == GATE_ID["SX"] or 
            gate.type() == GATE_ID["SY"] or
            gate.type() == GATE_ID["SW"] or
            gate.type() == GATE_ID["U2"] or 
            gate.type() == GATE_ID["U3"] or
            gate.type() == GATE_ID["RX"] or
            gate.type() == GATE_ID["RY"]
        ):
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_Matrix(gate)
            self._algorithm.Based_InnerProduct_targ(
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif (
            gate.type() == GATE_ID["S"] or
            gate.type() == GATE_ID["S_dagger"] or
            gate.type() == GATE_ID["RZ"] or
            gate.type() == GATE_ID["Phase"]
        ):
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_Matrix(gate)
            self._algorithm.Diagonal_Multiply_targ(
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["X"]:
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.RDiagonal_Swap_targ(
                t_index,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["Y"]:
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_Matrix(gate)
            self._algorithm.RDiagonal_MultiplySwap_targ(
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif (
            gate.type() == GATE_ID["Z"] or
            gate.type() == GATE_ID["U1"] or
            gate.type() == GATE_ID["T"] or
            gate.type() == GATE_ID["T_dagger"]
        ):
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_Matrix(gate)
            self._algorithm.Controlled_Multiply_targ(
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif (
            gate.type() == GATE_ID["CZ"] or
            gate.type() == GATE_ID["CU1"]
        ):
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            matrix = self.get_Matrix(gate)
            self._algorithm.Controlled_Multiply_ctargs(
                c_index,
                t_index,
                matrix,
                self._vector,
                self._qubits,
                8,
                self._sync
            )
        elif gate.type() == GATE_ID["CRz"]:
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            matrix = self.get_Matrix(gate)
            self._algorithm.Controlled_Multiply_ctargs(
                c_index,
                t_index,
                matrix,
                self._vector,
                self._qubits,
                12,
                self._sync
            )
        elif gate.type() == GATE_ID["RZZ"]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_Matrix(gate)
            self._algorithm.Diagonal_Multiply_targs(
                t_indexes,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif (
            gate.type() == GATE_ID["CX"] or
            gate.type() == GATE_ID["CY"]
        ):
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            matrix = self.get_Matrix(gate)
            self._algorithm.Controlled_MultiplySwap_ctargs(
                c_index,
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["CH"] or gate.type() == GATE_ID["CU3"]:
            t_index = self._qubits - 1 - gate.targ
            c_index = self._qubits - 1 - gate.carg
            matrix = self.get_Matrix(gate)
            self._algorithm.Controlled_InnerProduct_ctargs(
                c_index,
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["FSim"]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_Matrix(gate)
            self._algorithm.Completed_MxIP_targs(
                t_indexes,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["RXX"] or gate.type() == GATE_ID["RYY"]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_Matrix(gate)
            self._algorithm.Completed_IPxIP_targs(
                t_indexes,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["Swap"]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            self._algorithm.Controlled_Swap_targs(
                t_indexes,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["ID"]:
            pass
        elif gate.type() == 46 or gate.type() == GATE_ID["CCX"]:
            # TODO: GATE_ID["CCX"] = 30, not matched
            c_indexes = [self._qubits - 1 - carg for carg in gate.cargs]
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.Controlled_Swap_more(
                c_indexes,
                t_index,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["CCRz"]:
            c_indexes = [self._qubits - 1 - carg for carg in gate.cargs]
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_Matrix(gate)
            self._algorithm.Controlled_Multiply_more(
                c_indexes,
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["CSwap"]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            c_index = self._qubits - 1 - gate.carg
            self._algorithm.Controlled_Swap_tmore(
                t_indexes,
                c_index,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["Measure"]:
            index = self._qubits - 1 - gate.targ
            result = self._algorithm.MeasureGate_Apply(
                index,
                self._vector,
                self._qubits,
                self._sync
            )
            self.circuit.qubits[gate.targ].measured = result
        elif gate.type() == GATE_ID["Reset"]:
            # TODO: Not applied yet.
            index = self._qubits - 1 - gate.targ
            self._algorithm.ResetGate_Apply(
                index,
                self._vector,
                self._qubits,
                self._sync
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
            if gate.targets >= 12:
                pass
            else:
                self._algorithm.PermGate_Apply(
                    gate.pargs,
                    self._vector,
                    self._qubits,
                    self._sync
                )
        elif gate.type() == GATE_ID["PermFxT"]:
            self._algorithm.PermFxGate_Apply(
                gate.pargs,
                gate.targets,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["PermT"]:
            mapping = np.array(gate.pargs)
            self._algorithm.VectorPermutation(
                self.vector,
                mapping,
                changeInput=True,
                gpu_out=False,
                sync=self._sync
            )
        else:
            aux = cp.zeros_like(self._vector)
            matrix = self.get_Matrix(gate)
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

    def apply_combined_gates(self, gates):
        based_matrix = gates[0].compute_matrix
        t_index = self._qubits - 1 - gates[0].targ

        for gate in gates[1:]:
            based_matrix = np.dot(based_matrix, gate.compute_matrix)

        if self._precision == np.complex64:
            based_matrix = cp.array(based_matrix).astype(cp.complex64)
        else:
            based_matrix = cp.array(based_matrix)

        """
        H;
        self._algorithm.Based_InnerProduct_targ

        MSwap (odd)/ [Multiply/CMultiply with swap (odd)]
        self._algorithm.RDiagonal_MultiplySwap_targ

        Only Swap: (odd)
        self._algorithm.RDiagonal_Swap_targ

        Multiply/[Swap/MSwap (even)]
        self._algorithm.Diagonal_Multiply_targ
        
        CMultiply
        self._algorithm.Controlled_Multiply_targ
        """

        self._algorithm.Based_InnerProduct_targ(
            t_index,
            based_matrix,
            self._vector,
            self._qubits,
            self._sync
        )

    def apply_2qubits_gates(self, gates: list):
        matrix = np.kron(gates[1].compute_matrix, gates[0].compute_matrix)

        if self._precision == np.complex64:
            matrix = cp.array(matrix).astype(cp.complex64)
        else:
            matrix = cp.array(matrix)

        print(matrix)

        """
        gates from low - high
        """

        self._algorithm.Based_InnerProduct_targs(
            c_index = self._qubits - 1 - gates[0].targ,
            t_index = self._qubits - 1 - gates[1].targ,
            mat=matrix,
            vec=self._vector,
            vec_bit = self._qubits,
            sync=self._sync
        )
