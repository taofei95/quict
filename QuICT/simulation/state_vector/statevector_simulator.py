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
from QuICT.core.noise import NoiseModel
from QuICT.core.virtual_machine import VirtualQuantumMachine
from QuICT.simulation.utils import GateSimulator
from QuICT.tools.exception.core import TypeError
from QuICT.tools.exception.simulation import SampleBeforeRunError


class StateVectorSimulator:
    """ The simulator for qubits' vector state. """
    @property
    def circuit(self):
        return self._circuit

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vec):
        self._vector = self._gate_calculator.normalized_state_vector(vec, self._qubits)

    @property
    def device(self):
        return self._gate_calculator._gpu_device_id

    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double",
        gpu_device_id: int = 0,
        sync: bool = True
    ):
        """
        Args:
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
            precision (str, optional): The precision for the state vector, one of [single, double].
                Defaults to "double".
            gpu_device_id (int, optional): The GPU device ID. Defaults to 0.
            sync (bool, optional): Sync mode or Async mode. Defaults to True.
        """
        self._gate_calculator = GateSimulator(device, precision, gpu_device_id, sync)
        self._quantum_machine = None

    def initial_circuit(self, circuit: Circuit):
        """ Initial the qubits, quantum gates and state vector by given quantum circuit. """
        self._origin_circuit = circuit
        self._circuit = circuit if self._quantum_machine is None else self._quantum_machine.transpile(circuit)
        self._qubits = int(circuit.width())
        self._pipeline = circuit.fast_gates

        self._gate_calculator.gate_matrix_combined(self._circuit)

    def initial_state_vector(self, all_zeros: bool = False):
        """ Initial qubits' vector states. """
        if not all_zeros:
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits)
        else:
            self._vector = self._gate_calculator.get_empty_state_vector(self._qubits)

    def run(
        self,
        circuit: Circuit,
        quantum_state: np.ndarray = None,
        quantum_machine_model: Union[NoiseModel, VirtualQuantumMachine] = None,
        use_previous: bool = False
    ) -> np.ndarray:
        """ start simulator with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            quantum_state (ndarray): The initial quantum state vector.
            quantum_machine_model (Union[NoiseModel, VirtualQuantumMachine]): The model of quantum machine
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Returns:
            Union[cp.array, np.array]: The state vector.
        """
        # Deal with the Physical Machine Model
        self._quantum_machine = None
        if quantum_machine_model is not None:
            noise_model = quantum_machine_model if isinstance(quantum_machine_model, NoiseModel) else \
                NoiseModel(quantum_machine_info=quantum_machine_model)
            if not noise_model.is_ideal_model():
                self._quantum_machine = noise_model

        # Initial Quantum Circuit and State Vector
        self.initial_circuit(circuit)
        self._original_state_vector = None
        if quantum_state is not None:
            self._vector = self._gate_calculator.normalized_state_vector(quantum_state.copy(), self._qubits)
            if self._quantum_machine is not None:
                self._original_state_vector = quantum_state.copy()
        elif not use_previous:
            self.initial_state_vector()

        # Apply gates one by one
        self._run()

        return self.vector

    def _run(self):
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
                raise TypeError("StateVectorSimulation.run.circuit", "[CompositeGate, BasicGate, Trigger]". type(gate))

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
            self._pipeline = self._pipeline[:position] + mapping_cgate.fast_gates + \
                self._pipeline[position:]

    def _apply_measure_gate(self, qidx):
        result = self._gate_calculator.apply_measure_gate(qidx, self._vector, self._qubits)
        if self._quantum_machine is not None:
            result = self._quantum_machine.apply_readout_error(qidx, result)

        self._origin_circuit.qubits[self._qubits - 1 - qidx].measured = int(result)

    def _apply_reset_gate(self, qidx):
        self._gate_calculator.apply_reset_gate(qidx, self._vector, self._qubits)

    # TODO: refactoring later, multi-gpu kernel function
    def apply_multiply(self, value: Union[float, complex]):
        """ Deal with Operator <Multiply>

        Args:
            value (Union[float, complex]): The multiply value apply to state vector.
        """
        from QuICT.ops.gate_kernel.multigpu import float_multiply, complex_multiply

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
        if self._quantum_machine is not None:
            return self._sample_with_noise(shots, target_qubits)

        target_qubits = target_qubits if target_qubits is not None else list(range(self._qubits))
        state_list = [0] * (1 << len(target_qubits))
        original_sv = self._vector.copy()
        for _ in range(shots):
            for m_id in target_qubits:
                index = self._qubits - 1 - m_id
                self._apply_measure_gate(index)

            state_list[int(self._circuit.qubits[target_qubits])] += 1
            self._vector = original_sv.copy()

        return state_list

    def _sample_with_noise(self, shots: int, target_qubits: list) -> list:
        target_qubits = target_qubits if target_qubits is not None else list(range(self._qubits))
        state_list = [0] * (1 << len(target_qubits))

        for _ in range(shots):
            final_state = 0
            for m_id in target_qubits:
                index = self._qubits - 1 - m_id
                measured = self._gate_calculator.apply_measure_gate(index, self._vector, self._qubits)
                final_state <<= 1
                final_state += measured

            # Apply readout noise
            final_state = self._quantum_machine.apply_readout_error(target_qubits, final_state)
            state_list[final_state] += 1

            # Re-generate noised circuit and initial state vector
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits) \
                if self._original_state_vector is None else self._original_state_vector.copy()
            noised_circuit = self._quantum_machine.transpile(self._origin_circuit)
            self._pipeline = noised_circuit.fast_gates
            self._gate_calculator.gate_matrix_combined(noised_circuit)
            self._run()

        return state_list
