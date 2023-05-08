from numpy.random import random
from typing import Union, List
from collections import defaultdict

from .noise_error import QuantumNoiseError, NoiseChannel, BitflipError
from .readout_error import ReadoutError
from QuICT.core import Circuit
from QuICT.core.gate import BasicGate, GateType, GATEINFO_MAP, Unitary
from QuICT.core.operator import NoiseGate
from QuICT.core.virtual_machine import VirtualQuantumMachine
from QuICT.tools.exception.core import TypeError, ValueError, NoiseApplyError


class NoiseModel:
    """
    The Noise Model, which contains the QuantumNoiseErrors and can translated circuit with the given QuantumNoiseError.
    """
    def __init__(
        self,
        name: str = "Noise Model",
        quantum_machine_info: VirtualQuantumMachine = None
    ):
        """
        Args:
            name (str, optional): The name of this NoiseModel. Defaults to "Noise Model".
            quantum_machine_info (VirtualQuantumMachine, optional): The information about the quantum machine,
                can use those info to build NoiseModel for target machine. Defaults to None.
        """
        self._name = name
        self._vqm = quantum_machine_info

        self._error_by_gate = defaultdict(list)
        self._readout_errors = []

        if self._vqm is not None:
            self._build_nm_from_vqm()

    def is_ideal_model(self) -> bool:
        """ Validate it is a ideal(No noise) Quantum Machine Model or not. """
        if len(self._error_by_gate) + len(self._readout_errors) == 0:
            return True

        return False

    def _build_nm_from_vqm(self):
        qureg = self._vqm.qubits
        iset = self._vqm.instruction_set

        # Build Error from the given Qubits Fidelity. TODO: add QSP fidelity Error later
        for idx, qubit in enumerate(qureg):
            # Readout Error
            measured_fidelity = qubit.fidelity
            if isinstance(measured_fidelity, tuple):
                f0, f1 = measured_fidelity
            else:
                f0, f1 = measured_fidelity, measured_fidelity

            if f0 != 1.0 or f1 != 1.0:
                readout_error = ReadoutError([[f0, 1 - f0], [1 - f1, f1]])
                self.add_readout_error(readout_error, idx)

            # Gate Error
            gate_fidelity = qubit.gate_fidelity
            target_gate_type = iset.gates[:-1] if isinstance(gate_fidelity, float) else gate_fidelity.keys()
            for gate_type in target_gate_type:
                current_fidelity = gate_fidelity if isinstance(gate_fidelity, float) else gate_fidelity[gate_type]
                if current_fidelity != 1.0:
                    self.add(BitflipError(current_fidelity), gate_type.name, idx)

        # Deal with bi-qubits gate fidelity (coupling strength)
        coupling_strength = qureg._original_coupling_strength
        if coupling_strength is not None:
            bi_qubits_gate = iset.gates[-1]
            for start, end, val in coupling_strength:
                noise_error = BitflipError(val)
                noise_error = noise_error.tensor(noise_error)
                self.add(noise_error, bi_qubits_gate.name, [start, end])

    def __str__(self):
        nm_str = f"{self._name}:\nBasic Gates: {self._error_by_gate.keys()}\nNoise Errors:\n"

        for gate_type, noise_list in self._error_by_gate.items():
            nm_str += f"Gate {gate_type} with noise: \n"
            for noise, qubits in noise_list:
                if qubits == -1:
                    nm_str += f"target qubits: All, noise type: {noise.type.name};\n"
                else:
                    nm_str += f"target qubits: {qubits}, noise type: {noise.type.name};\n"

        if self._readout_errors:
            nm_str += "Readout Error: \n"
            for re_error, re_qubit in self._readout_errors:
                nm_str += f"target qubits: {re_qubit}, Readout Error Prob: {re_error.prob}; \n"

        return nm_str

    def _qubits_normalize(self, qubits: Union[int, List[int]]):
        if isinstance(qubits, int):
            qubits = [qubits]
        elif not isinstance(qubits, (list, tuple)):
            raise TypeError("NoiseModel.add.qubits", "int/list<int>", type(qubits))

        for q in qubits:
            if q < 0 or not isinstance(q, int):
                raise ValueError("NoiseModel.add.qubits", "be a positive integer", q)

        return qubits

    def _gates_normalize(self, noise: QuantumNoiseError, gates: Union[str, List[str]]):
        assert isinstance(noise, QuantumNoiseError), \
            TypeError("NoiseModel.add.noise", "QuantumNoiseError", type(noise))

        if isinstance(gates, str):
            gates = [gates]
        elif not isinstance(gates, (list, tuple)):
            raise TypeError("NoiseModel.add.gates", "str/list<str>", type(gates))

        for g in gates:
            assert isinstance(g, str), TypeError("NoiseModel.add.gates", "str/list<str>", type(g))

            gate_info = GATEINFO_MAP[GateType.__members__[g]]
            if gate_info[0] + gate_info[1] != noise.qubits:
                raise NoiseApplyError(
                    f"Un-matched qubits number between gate {gate_info[0] + gate_info[1]}" +
                    f"with noise error {noise.qubits}."
                )

        return gates

    def add(self, noise: QuantumNoiseError, gates: Union[str, List[str]], qubits: Union[int, List[int]] = None):
        """ Add noise which will affect target quantum gates with special qubits in the circuit.

        Args:
            noise (QuantumNoiseError): The noise error.
            gates (Union[str, List[str]]): The affected quantum gates.
            qubits (Union[int, List[int]], optional): The affected qubits, if None, same as add_noise_for_all_qubits.
                Defaults to None.
        """
        if qubits is None:
            self.add_noise_for_all_qubits(noise, gates)
            return

        # Input normalization
        qubits = self._qubits_normalize(qubits)
        gates = self._gates_normalize(noise, gates)

        # Add noise in the NoiseModel
        for gate in gates:
            self._error_by_gate[gate].append((noise, qubits))

    def add_noise_for_all_qubits(self, noise: QuantumNoiseError, gates: Union[str, List[str]]):
        """ Add noise which will affect all qubits in the circuit.

        Args:
            noise (QuantumNoiseError): The noise error
            gates (Union[str, List[str]]): The affected quantum gates.
        """
        # Input normalization
        gates = self._gates_normalize(noise, gates)

        # Add noise in the NoiseModel
        for gate in gates:
            self._error_by_gate[gate].append((noise, -1))

    def add_readout_error(self, noise: ReadoutError, qubits: Union[int, List[int]]):
        """ Add Readout error in the noise model.

        Example:
            1. Add 1-qubit Readout Error for all target qubits.
                noisemodel.add_readout_error(1-qubit ReadoutError, [0, 1, 2, etc])
            2. Add multi-qubits Readout Error for specified qubits (the number of qubits equal to ReadoutError.qubits).
                noisemodel.add_readout_error(multi-qubits ReadoutError, [1, 3, etc])

        Important:
            Supports to apply more than one Readout Error with the same target qubits, but do not support the
            combination of the Readout Error with the different qubits currently.

        Args:
            noise (ReadoutError): The Readout Error.
            qubits (Union[int, List[int]]): The target qubits for the Readout error
        """
        assert isinstance(noise, ReadoutError), TypeError("NoiseModel.addreadouterror", "ReadoutError", type(noise))
        qubits = self._qubits_normalize(qubits)

        if noise.qubits > 1:
            assert noise.qubits == len(qubits), NoiseApplyError(
                "For multi qubits Readout Error, the given qubits' number should equal to the noise's qubit number."
            )

        self._readout_errors.append((noise, qubits))

    def apply_readout_error(self, qureg):
        """ Apply readout error to target qubits.

        Args:
            qureg (Qureg): The circuits' qubits.
        """
        for readouterror, qubits in self._readout_errors:
            if readouterror.qubits == 1:
                for q in qubits:
                    if qureg[q].measured is None:
                        continue

                    truly_result = int(qureg[q])
                    qureg[q].measured = readouterror.apply_to_qubits(truly_result)
            else:
                all_qubits_measured = True
                for q in qubits:
                    if qureg[q].measured is None:
                        all_qubits_measured = False
                        break

                if not all_qubits_measured:
                    continue

                truly_result = int(qureg[qubits])
                noised_result = readouterror.apply_to_qubits(truly_result)
                for q in qubits[::-1]:
                    qureg[q].measured = noised_result & 1
                    noised_result >>= 1

    def transpile(self, circuit: Circuit, accumulated_mode: bool = False) -> Circuit:
        """ Apply all noise in the Noise Model to the given circuit, replaced related gate with the NoiseGate

        Args:
            circuit (Circuit): The given circuit.
            accumulated_mode (bool): Whether using accumulated mode to generate Noise Circuit.

        Returns:
            Circuit: The noised circuit with.
        """
        qubit_num = circuit.width()
        noised_circuit = Circuit(qubit_num)
        for gate in circuit.gates:
            if isinstance(gate, BasicGate):
                gate_str = gate.type.name
                if gate_str not in self._error_by_gate.keys():
                    gate | noised_circuit
                    continue

                gate_args = gate.cargs + gate.targs
                noise_list = self._error_by_gate[gate_str]
                append_origin_gate = True
                for noise, qubits in noise_list:
                    if qubits == -1 or (set(qubits) & set(gate_args)) == set(gate_args):    # noise's qubit matched
                        if accumulated_mode or noise.type == NoiseChannel.damping:
                            NoiseGate(gate, noise) & gate_args | noised_circuit
                            append_origin_gate = False
                        else:
                            prob = random()
                            noise_matrix = noise.prob_mapping_operator(prob)
                            Unitary(noise_matrix) | noised_circuit(gate_args)

                if append_origin_gate:
                    gate | noised_circuit
            else:
                gate | noised_circuit

        return noised_circuit
