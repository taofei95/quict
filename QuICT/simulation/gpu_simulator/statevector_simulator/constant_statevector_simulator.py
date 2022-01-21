#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/28 下午4:50
# @Author  : Kaiqi Li
# @File    : constant_statevecto_simulator

from collections import defaultdict
import numpy as np
import cupy as cp

from QuICT.core import Circuit
from QuICT.core.gate import Measure
from QuICT.core.utils import GateType
from QuICT.ops.utils import LinAlgLoader
from QuICT.simulation.gpu_simulator import BasicGPUSimulator
from QuICT.simulation.optimization import Optimizer
from QuICT.simulation.utils import GATE_TYPE_to_ID, GateGroup


class ConstantStateVectorSimulator(BasicGPUSimulator):
    """
    The simulator for qubits' vector state.

    Args:
        circuit (Circuit): The quantum circuit.
        precision (str): The precision for the state vector, single precision means complex64,
            double precision means complex128.
        gpu_device_id (int): The GPU device ID.
        sync (bool): Sync mode or Async mode.
    """
    def __init__(
        self,
        precision: str = "double",
        optimize: bool = False,
        gpu_device_id: int = 0,
        sync: bool = True
    ):
        BasicGPUSimulator.__init__(self, precision, gpu_device_id, sync)
        self._optimize = optimize

        if self._optimize:
            self._optimizor = Optimizer()

        # Initial simulator with limit_qubits
        self._algorithm = LinAlgLoader(device="GPU", enable_gate_kernel=True, enable_multigpu_gate_kernel=False)

    def _initial_circuit(self, circuit, use_previous):
        """ Initial the qubits, quantum gates and state vector by given quantum circuit. """
        self._circuit = circuit
        self._qubits = int(circuit.width())
        self._measure_result = defaultdict(list)

        if self._optimize:
            self._gates = self._optimizor.optimize(circuit.gates)
        else:
            self._gates = circuit.gates

        # Initial GateMatrix
        self._gate_matrix_prepare()

        # Initial vector state
        if not use_previous or self._vector is None:
            self._initial_vector_state()

    def _initial_vector_state(self):
        """ Initial qubits' vector states. """
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

    def run(
        self,
        circuit: Circuit,
        use_previous: bool = False,
        record_measured: bool = False
    ) -> np.ndarray:
        """ start simulator with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.
            record_measured (bool, optional): Record measured result within circuit or not.
        Returns:
            [array]: The state vector.
        """
        self._initial_circuit(circuit, use_previous)

        with cp.cuda.Device(self._device_id):
            for gate in self._gates:
                self.apply_gate(gate)

        if record_measured:
            return self.vector, self._measure_result
        else:
            return self.vector

    def apply_gate(self, gate):
        """ Depending on the given quantum gate, apply the target algorithm to calculate the state vector.

        Args:
            gate (Gate): the quantum gate in the circuit.
        """
        gate_type = gate.type
        default_parameters = (self._vector, self._qubits, self._sync)

        # [H, SX, SY, SW, U2, U3, Rx, Ry]
        if gate_type in GATE_TYPE_to_ID[GateGroup.matrix_1arg]:
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Based_InnerProduct_targ(
                t_index,
                matrix,
                *default_parameters
            )
        # [RZ, Phase]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.diagonal_1arg]:
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Diagonal_Multiply_targ(
                t_index,
                matrix,
                *default_parameters
            )
        # [X]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.swap_1arg]:
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.RDiagonal_Swap_targ(
                t_index,
                *default_parameters
            )
        # [Y]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.reverse_1arg]:
            t_index = self._qubits - 1 - gate.targ
            matrix = self.get_gate_matrix(gate)
            self._algorithm.RDiagonal_MultiplySwap_targ(
                t_index,
                matrix,
                *default_parameters
            )
        # [Z, U1, T, T_dagger, S, S_dagger]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.control_1arg]:
            t_index = self._qubits - 1 - gate.targ
            val = gate.matrix[1, 1]
            self._algorithm.Controlled_Multiply_targ(
                t_index,
                val,
                *default_parameters
            )
        # [CRz]
        elif gate_type == GateType.crz:
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
        elif gate_type in GATE_TYPE_to_ID[GateGroup.control_2arg]:
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
        elif gate_type in GATE_TYPE_to_ID[GateGroup.diagonal_2arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Diagonal_Multiply_targs(
                t_indexes,
                matrix,
                *default_parameters
            )
        # [CX, CY]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.reverse_2arg]:
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
        elif gate_type in GATE_TYPE_to_ID[GateGroup.matrix_2arg]:
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
        elif gate_type in GATE_TYPE_to_ID[GateGroup.complexMIP_2arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Completed_MxIP_targs(
                t_indexes,
                matrix,
                *default_parameters
            )
        # [Rxx, Ryy]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.complexIPIP_2arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Completed_IPxIP_targs(
                t_indexes,
                matrix,
                *default_parameters
            )
        # [Swap]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.swap_2arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            self._algorithm.Controlled_Swap_targs(
                t_indexes,
                *default_parameters
            )
        # [ID]
        elif gate_type == GateType.id:
            pass
        # [CCX]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.reverse_3arg]:
            c_indexes = [self._qubits - 1 - carg for carg in gate.cargs]
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.Controlled_Swap_more(
                c_indexes,
                t_index,
                *default_parameters
            )
        # [CCRz]
        elif gate_type in GATE_TYPE_to_ID[GateGroup.control_3arg]:
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
        elif gate_type in GATE_TYPE_to_ID[GateGroup.swap_3arg]:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            c_index = self._qubits - 1 - gate.carg
            self._algorithm.Controlled_Swap_tmore(
                t_indexes,
                c_index,
                *default_parameters
            )
        # [Measure]
        elif gate_type == GateType.measure:
            index = self._qubits - 1 - gate.targ
            result = self._algorithm.MeasureGate_Apply(
                index,
                *default_parameters
            )
            self.circuit.qubits[gate.targ].measured = result
            self._measure_result[index].append(result)
        # [Reset]
        elif gate_type == GateType.reset:
            index = self._qubits - 1 - gate.targ
            self._algorithm.ResetGate_Apply(
                index,
                *default_parameters
            )
        # [Barrier]
        elif gate_type == GateType.barrier:
            pass
        elif gate_type == GateType.perm:
            args = gate.cargs + gate.targs
            if len(args) == self._qubits:
                mapping = np.array(gate.pargs, dtype=np.int32)
            else:
                mapping = np.arange(self._qubits, dtype=np.int32)
                for idx, parg in enumerate(gate.pargs):
                    mapping[args[idx]] = args[parg]

            self._algorithm.VectorPermutation(
                self._vector,
                mapping,
                changeInput=True,
                gpu_out=False,
                sync=self._sync
            )
        elif gate_type in GATE_TYPE_to_ID[GateGroup.perm_gate]:
            args = gate.cargs + gate.targs
            if len(args) == self._qubits:
                self._algorithm.simple_vp(
                    self._vector,
                    np.array(gate.pargs, dtype=np.int32),
                    changeInput=True,
                    gpu_out=False,
                    sync=self._sync
                )
            else:
                self._algorithm.qubit_vp(
                    self._vector,
                    np.array(args, dtype=np.int32),
                    np.array(gate.pargs, dtype=np.int32),
                    changeInput=True,
                    gpu_out=False,
                    sync=self._sync
                )
        # [Unitary]
        elif gate_type == GateType.unitary:
            qubit_idxes = gate.cargs + gate.targs
            if len(qubit_idxes) == 1:   # 1-qubit unitary gate
                t_index = self._qubits - 1 - qubit_idxes[0]
                matrix = self.get_gate_matrix(gate)
                if gate.is_diagonal():    # diagonal gate
                    self._algorithm.Diagonal_Multiply_targ(
                        t_index,
                        matrix,
                        *default_parameters
                    )
                else:   # non-diagonal gate
                    self._algorithm.Based_InnerProduct_targ(
                        t_index,
                        matrix,
                        *default_parameters
                    )
            elif len(qubit_idxes) == 2:     # 2-qubits unitary gate
                indexes = [self._qubits - 1 - index for index in qubit_idxes]
                indexes.sort()
                matrix = self.get_gate_matrix(gate)
                if gate.is_diagonal():        # diagonal gate
                    self._algorithm.Diagonal_Multiply_targs(
                        indexes,
                        matrix,
                        *default_parameters
                    )
                else:   # non-diagonal gate
                    self._algorithm.Based_InnerProduct_targs(
                        indexes,
                        matrix,
                        *default_parameters
                    )
            else:   # common unitary gate supported, but run slowly
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
        # unsupported quantum gates
        else:
            raise KeyError(f"Unsupported Gate: {gate_type}")

    def sample(self):
        assert (self._circuit is not None)
        temp_measure_circuit = Circuit(self._qubits)
        Measure | temp_measure_circuit

        self.run(temp_measure_circuit, use_previous=True)
        measured_qubits = int(temp_measure_circuit.qubits)

        return measured_qubits
