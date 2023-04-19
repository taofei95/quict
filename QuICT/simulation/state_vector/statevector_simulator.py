#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/28 下午4:50
# @Author  : Kaiqi Li
# @File    : constant_statevecto_simulator

from typing import Union
import numpy as np

from QuICT.core import Circuit
from QuICT.core.operator import Trigger
from QuICT.core.gate import BasicGate, CompositeGate
from QuICT.core.utils import GateType
from QuICT.simulation.utils import GateSimulator
from QuICT.tools.exception.core import ValueError, TypeError
from QuICT.tools.exception.simulation import SampleBeforeRunError


class StateVectorSimulator:
    """
    The simulator for qubits' vector state.

    Args:
        circuit (Circuit): The quantum circuit.
        precision (str): The precision for the state vector, one of [single, double]. Defaults to "double".
        gpu_device_id (int): The GPU device ID.
        matrix_aggregation (bool): Using quantum gate matrix's aggregation to optimize running speed.
        sync (bool): Sync mode or Async mode.
    """
    __DEVICE = ["CPU", "GPU"]
    __PRECISION = ["single", "double"]

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, circuit):
        self._circuit = circuit

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vec):
        self._vector = self._gate_calculator.normalized_state_vector(vec, self._qubits)

    @property
    def device(self):
        return self._device_id

    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double",
        gpu_device_id: int = 0,
        sync: bool = True
    ):
        if device not in self.__DEVICE:
            raise ValueError("StateVectorSimulation.device", "[CPU, GPU]", device)

        if precision not in self.__PRECISION:
            raise ValueError("StateVectorSimulation.precision", "[single, double]", precision)

        self._device = device
        self._precision = precision
        self._device_id = gpu_device_id
        self._sync = sync
        self._gate_calculator = GateSimulator(self._device, self._precision, self._device_id, self._sync)

    def initial_circuit(self, circuit: Circuit):
        """ Initial the qubits, quantum gates and state vector by given quantum circuit. """
        self._circuit = circuit
        self._qubits = int(circuit.width())
        self._pipeline = circuit.fast_gates

    def initial_state_vector(self, all_zeros: bool = False):
        """ Initial qubits' vector states. """
        if not all_zeros:
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits)
        else:
            self._vector = self._gate_calculator.get_empty_state_vector(self._qubits)

    def run(
        self,
        circuit: Circuit,
        state_vector: np.ndarray = None,
        use_previous: bool = False
    ) -> np.ndarray:
        """ start simulator with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            state_vector (ndarray): The initial state vector.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Returns:
            Union[cp.array, np.array]: The state vector.
        """
        self.initial_circuit(circuit)
        if state_vector is not None:
            self._vector = self._gate_calculator.normalized_state_vector(state_vector, self._qubits)
        elif not use_previous:
            self.initial_state_vector()

        idx = 0
        while idx < len(self._pipeline):
            gate, qidxes, _ = self._pipeline[idx]
            idx += 1

            if isinstance(gate, CompositeGate):
                self._apply_compositegate(gate, qidxes)
            elif isinstance(gate, BasicGate):
                self._apply_gate(gate, qidxes)
            elif isinstance(gate, Trigger):
                self._apply_trigger(gate, qidxes, idx)
            else:
                raise TypeError("StateVectorSimulation.run.circuit", "[BasicGate, Trigger]". type(gate))

        return self.vector

    def _apply_gate(self, gate: BasicGate, qidxes: list):
        """ Depending on the given quantum gate, apply the target algorithm to calculate the state vector.

        Args:
            gate (Gate): the quantum gate in the circuit.
        """
        gate_type = gate.type
        if gate_type == GateType.measure:
            self._apply_measure_gate(self._qubits - 1 - qidxes[0])
        elif gate_type == GateType.reset:
            self._apply_reset_gate(self._qubits - 1 - qidxes[0])
        else:
            self._gate_calculator.apply_gate(gate, qidxes, self._vector, self._qubits)

    def _apply_compositegate(self, gate: CompositeGate, qidxes: list):
        """ Depending on the given quantum gate, apply the target algorithm to calculate the state vector.

        Args:
            gate (Gate): the quantum gate in the circuit.
        """
        qidxes_mapping = {}
        cgate_qlist = gate.qubits
        for idx, cq in enumerate(cgate_qlist):
            qidxes_mapping[cq] = qidxes[idx]

        for cgate, cg_idx, _ in gate.fast_gates:
            real_qidx = [qidxes_mapping[idx] for idx in cg_idx]
            if isinstance(cgate, CompositeGate):
                self._apply_compositegate(cgate, real_qidx)
            else:
                self._apply_gate(cgate, real_qidx)

    def _apply_trigger(self, op: Trigger, qidxes: list, current_idx: int) -> CompositeGate:
        """ Deal with the Operator <Trigger>.

        Args:
            op (Trigger): The operator Trigger
            current_idx (int): the index of Trigger in Circuit
        """
        for targ in qidxes:
            index = self._qubits - 1 - targ
            self._apply_measure_gate(index)

        mapping_cgate = op.mapping(int(self._circuit[qidxes]))
        if isinstance(mapping_cgate, CompositeGate):
            # TODO: checkpoint update
            # Check for checkpoint
            cp = mapping_cgate.checkpoint
            position = current_idx if cp is None else self._circuit.find_position(cp)
            self._pipeline = self._pipeline[:position] + mapping_cgate.gates + \
                self._pipeline[position:]

    def _apply_measure_gate(self, qidx):
        result = self._gate_calculator.apply_measure_gate(qidx, self._vector, self._qubits)
        self._circuit.qubits[self._qubits - 1 - qidx].measured = int(result)

    def _apply_reset_gate(self, qidx):
        self._gate_calculator.apply_reset_gate(qidx, self._vector, self._qubits)

    # TODO: refactoring later, multi-gpu kernel function
    def apply_multiply(self, value: Union[float, np.complex]):
        """ Deal with Operator <Multiply>

        Args:
            value (Union[float, complex]): The multiply value apply to state vector.
        """
        from QuICT.ops.gate_kernel import float_multiply, complex_multiply

        default_parameters = (self._vector, self._qubits, self._sync)
        if isinstance(value, float):
            float_multiply(value, *default_parameters)
        else:
            if self._precision == np.complex64:
                value = np.complex64(value)

            complex_multiply(value, *default_parameters)

    def apply_zeros(self):
        """ Set state vector to be zero. """
        self._vector = self._gate_calculator.get_empty_state_vector(self._qubits)

    def sample(self, shots: int = 1, target_qubits: list = None) -> list:
        """ Sample the measured result from current state vector, please first run simulator.run().

        **WARNING**: Please make sure the target qubits are not been measured before simulator.sample().

        Args:
            shots (int): The sample times for current state vector.
            target_qubits (List[int]): The indexes of qubits which want to be measured. If it is None, there
            will measured all qubits in previous circuits.

        Returns:
            List[int]: The measured result list with length equal to 2 ** len(target_qubits)
        """
        assert (self._circuit is not None), \
            SampleBeforeRunError("StateVectorSimulation sample without run any circuit.")
        original_sv = self._vector.copy()
        if target_qubits is None:
            target_qubits = list(range(self._qubits))

        state_list = [0] * (1 << len(target_qubits))

        for _ in range(shots):
            for m_id in target_qubits:
                index = self._qubits - 1 - m_id
                self._apply_measure_gate(index)

            state_list[int(self._circuit.qubits[target_qubits])] += 1
            self._vector = original_sv.copy()

        return state_list
