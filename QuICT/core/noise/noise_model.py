from numpy.random import random
from typing import Union, List

from .noise_error import QuantumNoiseError, NoiseChannel, BitflipError
from .readout_error import ReadoutError
from QuICT.core import Circuit
from QuICT.core.gate import BasicGate, GateType, gate_builder, Unitary
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

        self._gate_type = []
        self._error_by_gate = {}
        self._all_qubits_error_gates = {}
        self._readout_errors = []

        if self._vqm is not None:
            self._build_nm_from_vqm()

    def _build_nm_from_vqm(self):
        qureg = self._vqm.qubits
        iset = self._vqm.instruction_set

        # Build gate error from Instruction Set
        self._gate_type = iset.gates
        gate_fidelity = self._vqm.gate_fidelity
        if gate_fidelity is not None:
            for gate_type, fidelity in iset.one_qubit_fidelity.items():
                gate_error = BitflipError(fidelity)
                self.add_noise_for_all_qubits(gate_error, gate_type)

        # Build Readout Error from the given Qureg. TODO: add QSP fidelity Error later
        for idx, qubit in enumerate(qureg):
            if qubit.fidelity != 1.0:
                readout_error = ReadoutError([[qubit.fidelity, 1 - qubit.fidelity], [1 - qubit.fidelity, qubit.fidelity]])
                self.add_readout_error(readout_error, idx)

        # Deal with bi-qubits gate fidelity (coupling strength)
        coupling_strength = self._vqm.coupling_strength
        if coupling_strength is not None:
            bi_qubits_gate = self._gate_type[-1]
            for start, end, val in coupling_strength:
                noise_error = BitflipError(val)
                noise_error = noise_error.tensor(noise_error)
                self.add(noise_error, bi_qubits_gate, [start, end])

    def __str__(self):
        nm_str = f"{self._name}:\nBasic Gates: {self._gate_type}\nNoise Errors:\n"
        if self._all_qubits_error_gates:
            nm_str += f"Gates with all qubits: {set(self._all_qubits_error_gates)}\n"

        if self._specified_error_gates:
            nm_str += "Gates with specified qubits:"
            for _, q, g in self._instruction["specific"]:
                nm_str += f"[qubits: {q}, gates: {g}] "

        if self._readout_errors:
            nm_str += "Measured Gates with Readout Error:"
            for idx, re_qubit in enumerate(self._readout_errors):
                readout_error = self._instruction["readout"][idx]
                nm_str += f"[qubits: {re_qubit}, Readout Error Prob: {readout_error.prob}]"

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
        if not gates:
            raise ValueError("NoiseModel.add.gates", "not be empty", gates)

        if isinstance(gates, str):
            gates = [gates]
        elif not isinstance(gates, (list, tuple)):
            raise TypeError("NoiseModel.add.gates", "str/list<str>", type(gates))

        for g in gates:
            if g not in self._basic_gates:
                raise ValueError("NoiseModel.add.gates", self._basic_gates, g)

            gate = gate_builder(GateType.__members__[g])
            if gate.controls + gate.targets != noise.qubits:
                raise NoiseApplyError(
                    f"Un-matched qubits number between gate {gate.controls + gate.targets}" +
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
            if isinstance(gate, BasicGate) and gate.type.name in self._basic_gates:
                gate_str = gate.type.name
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
