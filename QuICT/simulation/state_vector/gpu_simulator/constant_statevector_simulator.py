#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/28 下午4:50
# @Author  : Kaiqi Li
# @File    : constant_statevecto_simulator

from collections import defaultdict
from copy import deepcopy
import numpy as np
import cupy as cp

from QuICT.core import Circuit
from QuICT.core.operator import Trigger
from QuICT.core.gate import Measure, BasicGate, CompositeGate
from QuICT.core.utils import GateType, MatrixType
from QuICT.ops.utils import LinAlgLoader
from QuICT.simulation.optimization import Optimizer
from QuICT.ops.gate_kernel import Float_Multiply, Simple_Multiply
from QuICT.simulation.utils import GateMatrixs


class ConstantStateVectorSimulator:
    """
    The simulator for qubits' vector state.

    Args:
        circuit (Circuit): The quantum circuit.
        precision (str): The precision for the state vector, single precision means complex64,
            double precision means complex128.
        gpu_device_id (int): The GPU device ID.
        sync (bool): Sync mode or Async mode.
    """
    __PRECISION = ["single", "double"]

    @property
    def circuit(self):
        return self._circuit

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vec):
        with cp.cuda.Device(self._device_id):
            if type(vec) is np.ndarray:
                self._vector = cp.array(vec)
            else:
                self._vector = vec

    @property
    def device(self):
        return self._device_id

    def __init__(
        self,
        precision: str = "double",
        gpu_device_id: int = 0,
        optimize: bool = False,
        matrix_aggregation: bool = True,
        sync: bool = True
    ):
        if precision not in self.__PRECISION:
            raise ValueError("Wrong precision. Please use one of [single, double].")

        self._precision = np.complex128 if precision == "double" else np.complex64
        self._device_id = gpu_device_id
        self._sync = sync

        self._using_matrix_aggregation = matrix_aggregation
        self._optimize = optimize

        # Initial simulator with limit_qubits
        self._algorithm = LinAlgLoader(device="GPU", enable_gate_kernel=True, enable_multigpu_gate_kernel=False)
        # Set gpu id
        cp.cuda.runtime.setDevice(self._device_id)

    def initial_circuit(self, circuit: Circuit):
        """ Initial the qubits, quantum gates and state vector by given quantum circuit. """
        self._circuit = circuit
        self._qubits = int(circuit.width())
        self._measure_result = defaultdict(list)
        self._pipeline = []

        if self._optimize:
            self._pipeline = Optimizer().optimize(circuit.gates)
        else:
            for gate in circuit.gates:
                if isinstance(gate, BasicGate):
                    self._pipeline.append(deepcopy(gate))
                else:
                    self._pipeline.append(gate)

        # Initial GateMatrix if matrix_aggregation is True
        if self._using_matrix_aggregation:
            self._initial_matrix_aggregation()

    def _initial_matrix_aggregation(self):
        # Pretreatment gate matrixs optimizer
        self.gateM_optimizer = GateMatrixs(self._precision, self._device_id)
        self.gateM_optimizer.build(self._pipeline)

    def get_gate_matrix(self, gate: BasicGate):
        """ Return the gate's matrix in GPU. """
        if self._using_matrix_aggregation:
            return self.gateM_optimizer.get_target_matrix(gate)
        else:
            return cp.array(gate.matrix, dtype=self._precision)

    def initial_state_vector(self, qubits: int = 0, all_zeros: bool = False):
        """ Initial qubits' vector states. """
        if qubits != 0:
            self._qubits = qubits

        vector_size = 1 << int(self._qubits)
        self._vector = cp.zeros(vector_size, dtype=self._precision)
        if not all_zeros:
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
        self.initial_circuit(circuit)
        if not use_previous:
            self.initial_state_vector()

        idx = 0
        while self._pipeline:
            gate = self._pipeline.pop(0)
            if isinstance(gate, BasicGate):
                self.apply_gate(gate)
            elif isinstance(gate, Trigger):
                mapping_cgate = self.apply_trigger(gate)
                if mapping_cgate is not None:
                    # optimized composite gate's matrix
                    if self._using_matrix_aggregation:
                        self.gateM_optimizer.build(mapping_cgate.gates)
                    # Check for checkpoint
                    cp = mapping_cgate.checkpoint
                    position = 0 if cp is None else self._circuit.find_position(cp) - idx
                    self._pipeline = self._pipeline[:position] + deepcopy(mapping_cgate.gates) + \
                        self._pipeline[position:]

            idx += 1

        if record_measured:
            return self.vector, self._measure_result
        else:
            return self.vector

    def apply_normal_matrix(self, gate: BasicGate):
        # Get gate's parameters
        assert gate.matrix_type == MatrixType.normal
        args_num = gate.controls + gate.targets
        gate_args = gate.cargs + gate.targs
        matrix = self.get_gate_matrix(gate)
        default_parameters = (self._vector, self._qubits, self._sync)

        if args_num == 1:
            index = self._qubits - 1 - gate_args[0]
            self._algorithm.Based_InnerProduct_targ(
                index,
                matrix,
                *default_parameters
            )
        elif args_num == 2:
            if gate.controls == gate.targets:
                c_index = self._qubits - 1 - gate.carg
                t_index = self._qubits - 1 - gate.targ
                self._algorithm.Controlled_InnerProduct_ctargs(
                    c_index,
                    t_index,
                    matrix,
                    *default_parameters
                )
            elif gate.targets == 2:
                indexes = [self._qubits - 1 - index for index in gate_args]
                self._algorithm.Based_InnerProduct_targs(
                    indexes,
                    matrix,
                    *default_parameters
                )
            else:
                raise KeyError("Quantum gate cannot only have control qubits.")

    def apply_diagonal_matrix(self, gate: BasicGate):
        # Get gate's parameters
        assert gate.matrix_type in [MatrixType.diagonal, MatrixType.diag_diag]
        args_num = gate.controls + gate.targets
        gate_args = gate.cargs + gate.targs
        matrix = self.get_gate_matrix(gate)
        default_parameters = (self._vector, self._qubits, self._sync)

        if args_num == 1:
            index = self._qubits - 1 - gate.targ
            self._algorithm.Diagonal_Multiply_targ(
                index,
                matrix,
                *default_parameters
            )
        elif args_num == 2:
            if gate.controls == gate.targets:
                c_index = self._qubits - 1 - gate.carg
                t_index = self._qubits - 1 - gate.targ
                self._algorithm.Controlled_Multiply_ctargs(
                    c_index,
                    t_index,
                    matrix,
                    *default_parameters
                )
            elif gate.targets == 2:
                indexes = [self._qubits - 1 - index for index in gate_args]
                self._algorithm.Diagonal_Multiply_targs(
                    indexes,
                    matrix,
                    *default_parameters
                )
            else:
                raise KeyError("Quantum gate cannot only have control qubits.")
        else:   # [CCRz]
            c_indexes = [self._qubits - 1 - carg for carg in gate.cargs]
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.Controlled_Multiply_more(
                c_indexes,
                t_index,
                matrix,
                *default_parameters
            )

    def apply_swap_matrix(self, gate: BasicGate):
        # Get gate's parameters
        assert gate.matrix_type == MatrixType.swap
        args_num = gate.controls + gate.targets
        gate_args = gate.cargs + gate.targs
        default_parameters = (self._vector, self._qubits, self._sync)

        if args_num == 1:
            index = self._qubits - 1 - gate.targ
            self._algorithm.RDiagonal_Swap_targ(
                index,
                *default_parameters
            )
        elif args_num == 2:
            t_indexes = [self._qubits - 1 - targ for targ in gate_args]
            self._algorithm.Controlled_Swap_targs(
                t_indexes,
                *default_parameters
            )
        else:   # CSwap
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            c_index = self._qubits - 1 - gate.carg
            self._algorithm.Controlled_Swap_tmore(
                t_indexes,
                c_index,
                *default_parameters
            )

    def apply_reverse_matrix(self, gate: BasicGate):
        # Get gate's parameters
        assert gate.matrix_type == MatrixType.reverse
        args_num = gate.controls + gate.targets
        gate_args = gate.cargs + gate.targs
        matrix = self.get_gate_matrix(gate)
        default_parameters = (self._vector, self._qubits, self._sync)

        if args_num == 1:
            index = self._qubits - 1 - gate_args[0]
            self._algorithm.RDiagonal_MultiplySwap_targ(
                index,
                matrix,
                *default_parameters
            )
        elif args_num == 2:   # only consider 1 control qubit + 1 target qubit
            c_index = self._qubits - 1 - gate_args[0]
            t_index = self._qubits - 1 - gate_args[1]
            self._algorithm.Controlled_MultiplySwap_ctargs(
                c_index,
                t_index,
                matrix,
                *default_parameters
            )
        else:   # CCX
            c_indexes = [self._qubits - 1 - carg for carg in gate.cargs]
            t_index = self._qubits - 1 - gate.targ
            self._algorithm.Controlled_Swap_more(
                c_indexes,
                t_index,
                *default_parameters
            )

    def apply_control_matrix(self, gate: BasicGate):
        # Get gate's parameters
        assert gate.matrix_type == MatrixType.control
        args_num = gate.controls + gate.targets
        gate_args = gate.cargs + gate.targs
        default_parameters = (self._vector, self._qubits, self._sync)

        if args_num == 1:
            index = self._qubits - 1 - gate_args[0]
            val = gate.matrix[1, 1]
            self._algorithm.Controlled_Multiply_targ(
                index,
                val,
                *default_parameters
            )
        elif args_num == 2:
            c_index = self._qubits - 1 - gate_args[0]
            t_index = self._qubits - 1 - gate_args[1]
            val = gate.matrix[3, 3]
            self._algorithm.Controlled_Product_ctargs(
                c_index,
                t_index,
                val,
                *default_parameters
            )

    def apply_gate(self, gate: BasicGate):
        """ Depending on the given quantum gate, apply the target algorithm to calculate the state vector.

        Args:
            gate (Gate): the quantum gate in the circuit.
        """
        matrix_type = gate.matrix_type
        gate_type = gate.type
        default_parameters = (self._vector, self._qubits, self._sync)

        if gate_type == GateType.id or gate_type == GateType.barrier:
            return

        # Deal with quantum gate with more than 3 qubits.
        if (
            gate_type in [GateType.unitary, GateType.qft, GateType.iqft] and
            gate.targets >= 3
        ):
            aux = cp.zeros_like(self._vector)
            self._algorithm.matrix_dot_vector(
                gate.matrix,
                gate.controls + gate.targets,
                self._vector,
                self._qubits,
                gate.cargs + gate.targs,
                aux,
                self._sync
            )
            self.vector = aux
            return

        # [H, SX, SY, SW, U2, U3, Rx, Ry] 2-bits [CH, ] 2-bits[targets] [unitary]
        if matrix_type == MatrixType.normal:
            self.apply_normal_matrix(gate)
        # [Rz, Phase], 2-bits [CRz], 3-bits [CCRz]
        elif matrix_type in [MatrixType.diagonal, MatrixType.diag_diag]:
            self.apply_diagonal_matrix(gate)
        # [X] 2-bits [swap] 3-bits [CSWAP]
        elif matrix_type == MatrixType.swap:
            self.apply_swap_matrix(gate)
        # [Y] 2-bits [CX, CY] 3-bits: [CCX]
        elif matrix_type == MatrixType.reverse:
            self.apply_reverse_matrix(gate)
        # [S, sdg, Z, U1, T, tdg] # 2-bits [CZ, CU1]
        elif matrix_type == MatrixType.control:
            self.apply_control_matrix(gate)
        # [FSim]
        elif matrix_type == MatrixType.ctrl_normal:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Completed_MxIP_targs(
                t_indexes,
                matrix,
                *default_parameters
            )
        # [Rxx, Ryy]
        elif matrix_type == MatrixType.normal_normal:
            t_indexes = [self._qubits - 1 - targ for targ in gate.targs]
            matrix = self.get_gate_matrix(gate)
            self._algorithm.Completed_IPxIP_targs(
                t_indexes,
                matrix,
                *default_parameters
            )
        # [Measure, Reset]
        elif gate_type in [GateType.measure, GateType.reset]:
            index = self._qubits - 1 - gate.targ
            prob = self.get_measured_prob(index).get()
            self.apply_specialgate(index, gate_type, prob)
        # [Perm]
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
        # unsupported quantum gates
        else:
            raise KeyError(f"Unsupported Gate: {gate_type}")

    def apply_trigger(self, op: Trigger) -> CompositeGate:
        """ Deal with the Operator <Trigger>. """
        state = 0
        for targ in op.targs:
            index = self._qubits - 1 - targ
            prob = self.get_measured_prob(index).get()
            result = self.apply_specialgate(index, GateType.measure, prob)
            state <<= 1
            state += result

        return op.mapping(state)

    def apply_multiply(self, value):
        """ Deal with Operator <Multiply>. """
        default_parameters = (self._vector, self._qubits, self._sync)
        if isinstance(value, float):
            Float_Multiply(value, *default_parameters)
        else:
            Simple_Multiply(value, *default_parameters)

    def apply_zeros(self):
        """ Set state vector to be zero. """
        self._vector = cp.zeros_like(self.vector)

    def get_measured_prob(self, index: int, all_measured: bool = False) -> cp.ndarray:
        """ Return the probability of measured state 1. """
        return self._algorithm.measured_prob_calculate(
            index,
            self._vector,
            self._qubits,
            all_measured=all_measured,
            sync=self._sync
        )

    def apply_specialgate(self, index: int, type: GateType, prob: float = None):
        """ Apply Measure/Reset gate in to simulator. """
        if type == GateType.measure:
            result = int(self._algorithm.MeasureGate_Apply(
                index,
                self._vector,
                self._qubits,
                prob,
                self._sync
            ))
            self._circuit.qubits[self._qubits - 1 - index].measured = result
            self._measure_result[self._qubits - 1 - index].append(result)
            return result
        elif type == GateType.reset:
            return self._algorithm.ResetGate_Apply(
                index,
                self._vector,
                self._qubits,
                prob,
                self._sync
            )

    def sample(self):
        """ Return the measured result from current state vector. """
        assert (self._circuit is not None)
        original_sv = self._vector.copy()
        temp_measure_circuit = Circuit(self._qubits)
        Measure | temp_measure_circuit

        self.run(temp_measure_circuit, use_previous=True)
        measured_qubits = int(temp_measure_circuit.qubits)

        self._vector = original_sv
        return measured_qubits
