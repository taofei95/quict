from typing import Union, List
from collections import defaultdict

from QuICT.core.noise import QuantumNoiseError
from QuICT.core.gate import BasicGate, GateType, GATE_TYPE_TO_CLASS
from .noise_gate import NoiseGate


class NoiseModel:
    def __init__(self, name: str = "Noise Model", basic_gates: List = GateType.__members__.keys()):
        self._name = name
        self._basic_gates = list(basic_gates)
        self._instruction = defaultdict(list)
        self._all_qubits_error_gates = []
        self._specified_error_gates = []

    def __str__(self):
        nm_str = f"{self._name}:\nBasic Gates: {self._basic_gates}\nNoise Errors:\n"
        if self._all_qubits_error_gates:
            nm_str += f"Gates with all qubits: {set(self._all_qubits_error_gates)}\n"

        if self._specified_error_gates:
            nm_str += "Gates with specified qubits:"
            for _, q, g in self._instruction["specific"]:
                nm_str += f"[qubits: {q}, gates: {g}] "

        return nm_str

    def _qubits_normalize(self, qubits: Union[int, List[int]]):
        if isinstance(qubits, int):
            qubits = [qubits]
        elif not isinstance(qubits, (list, tuple)):
            raise TypeError("The qubits must be interger of list[integer].")

        for q in qubits:
            if q < 0 or not isinstance(q, int):
                raise KeyError("The qubits must be positive integer.")

    def _gates_normalize(self, noise: QuantumNoiseError, gates: Union[str, List[str]]):
        assert isinstance(noise, QuantumNoiseError), "Unsupportted noise error here."
        if not gates:
            raise KeyError("Must specified quantum gates for noise model.")

        if isinstance(gates, str):
            gates = [gates]
        elif not isinstance(gates, (list, tuple)):
            raise TypeError("The gates must be string of gate of list[string].")

        for g in gates:
            if g not in self._basic_gates:
                raise KeyError(f"The gate is not in based gates. Please use one of {self._basic_gates}.")

            gate = GATE_TYPE_TO_CLASS[GateType.__members__[g]]()
            if gate.controls + gate.targets != noise.qubits:
                raise KeyError(
                    f"Un-matched qubits number between gate {gate.controls + gate.targets}" +
                    f"with noise error {noise.qubits}."
                )

    def add(self, noise: QuantumNoiseError, gates: Union[str, List[str]], qubits: Union[int, List[int]] = None):
        if qubits is None:
            self.add_noise_for_all_qubits(noise, gates)
            return

        self._qubits_normalize(qubits)
        self._gates_normalize(noise, gates)

        self._instruction["specific"].append((noise, qubits, gates))
        self._specified_error_gates += gates if isinstance(gates, list) else [gates]

    def add_noise_for_all_qubits(self, noise: QuantumNoiseError, gates: Union[str, List[str]]):
        self._gates_normalize(noise, gates)
        self._instruction["all-qubits"].append((noise, gates))
        self._all_qubits_error_gates += gates if isinstance(gates, list) else [gates]

    def circuit_apply(self, circuit):
        cir_gates = circuit.gates[:]
        for idx, gate in enumerate(cir_gates):
            if not isinstance(gate, BasicGate):
                continue

            gate_str = gate.type.name
            noise_list = []
            if gate_str in self._all_qubits_error_gates:
                noise_list += self._kraus_matrix_for_all_qubits(gate)

            if gate_str in self._specified_error_gates:
                noise_list += self._kraus_matrix_for_specified_qubits(gate)

            if not noise_list:
                continue

            based_noise = noise_list[0]
            if len(noise_list) > 1:
                for n in noise_list[1:]:
                    based_noise = based_noise.compose(n)

            noise_gate = NoiseGate(gate, based_noise)
            circuit.replace_gate(idx, noise_gate)

        return circuit

    def _kraus_matrix_for_all_qubits(self, gate):
        gate_str = gate.type.name
        noise_list = []
        for noise, gate_list in self._instruction["all-qubits"]:
            if gate_str in gate_list:
                noise_list.append(noise)

        return noise_list

    def _kraus_matrix_for_specified_qubits(self, gate):
        gate_str = gate.type.name
        gate_idx = gate.cargs + gate.targs
        noise_list = []
        for noise, qubit, gate_list in self._instruction["specific"]:
            if isinstance(qubit, int):
                qubit_intersection = qubit in gate_idx
            else:
                qubit_intersection = (set(qubit) & set(gate_idx)) == set(qubit)

            if gate_str in gate_list and qubit_intersection:
                noise_list.append(noise)

        return noise_list
