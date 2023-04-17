from typing import Union

from QuICT.core import Qureg, Layout
from QuICT.core.noise import NoiseModel
from QuICT.qcda.synthesis import InstructionSet
from QuICT.tools.exception.core import TypeError, ValueError


class VirtualQuantumMachine:
    """ The Class store the information about Quantum Machine. """
    @property
    def qubit_number(self) -> int:
        return len(self._qubits)

    @property
    def qubits(self) -> Qureg:
        return self._qubits

    @qubits.setter
    def qubits(self, qubits: Qureg):
        assert isinstance(qubits, Qureg)
        self._qubits = qubits

    @property
    def instruction_set(self) -> InstructionSet:
        return self._instruction_set

    @instruction_set.setter
    def instruction_set(self, ins: InstructionSet):
        assert isinstance(ins, InstructionSet)
        self._instruction_set = ins

    @property
    def layout(self) -> Layout:
        return self._layout

    @layout.setter
    def layout(self, layout: Layout):
        assert isinstance(layout, Layout)
        assert layout.qubit_number == self.qubit_number
        self._layout = layout

    @property
    def qubit_fidelity(self) -> list:
        return self._qubit_fidelity

    @qubit_fidelity.setter
    def qubit_fidelity(self, qf: list):
        self._qubits.set_fidelity(qf)
        self._qubit_fidelity = qf

    @property
    def t1_time(self) -> list:
        return self._t1_times

    @t1_time.setter
    def t1_time(self, t1: list):
        self._qubits.set_t1_time(t1)
        self._t1_times = t1

    @property
    def t2_time(self) -> list:
        return self._t2_times

    @t2_time.setter
    def t2_time(self, t2: list):
        self._qubits.set_t2_time(t2)
        self._t2_times = t2

    @property
    def coupling_strength(self) -> dict:
        return self._coupling_strength

    @coupling_strength.setter
    def coupling_strength(self, cs: dict):
        self._qubits.set_couling_strength(cs)
        self._coupling_strength = cs

    @property
    def gate_fidelity(self) -> dict:
        return self._gate_fidelity
    
    @gate_fidelity.setter
    def gate_fidelity(self, gf: dict):
        assert isinstance(gf, dict)
        assert len(gf.keys()) == self._instruction_set.size()
        self._gate_fidelity = gf

    @property
    def noise_model(self) -> NoiseModel:
        return self._noise_model
    
    @noise_model.setter
    def noise_model(self, nm: NoiseModel):
        assert isinstance(nm, NoiseModel)
        self._noise_model = nm

    def __init__(
        self,
        qubits: Union[int, Qureg],
        instruction_set: InstructionSet,
        qubit_fidelity: list = None,
        t1_coherence_time: list = None,
        t2_coherence_time: list = None,
        coupling_strength: dict = None,
        layout: Layout = None,
        gate_fidelity: dict = None,
        noise_model: NoiseModel = None
    ):
        """
        Args:
            qubits (Union[int, Qureg]): The qubit number or the Qureg which is the list of Qubit.
            instruction_set (InstructionSet): The set of quantum gates which Quantum Machine supports.
            qubit_fidelity (list, optional): The fidelity for each qubit. Defaults to None.
            t1_coherence_time (list, optional): The t1 coherence time for each qubit. Defaults to None.
            t2_coherence_time (list, optional): The t2 coherence time for each qubit. Defaults to None.
            coupling_strength (dict, optional): The coupling strength between the qubits. Defaults to None.
            layout (Layout, optional): The description of physical topology of Quantum Machine. Defaults to None.
            gate_fidelity (dict, optional): The fidelity for each quantum gate. Defaults to None.
            noise_model (NoiseModel, optional): The noise model which describe the noise of Quantum Machine.
                Defaults to None.

        Raises:
            TypeError: The wrong type about input.
            ValueError: The illegal value of the given input.
        """
        # Describe the qubits of Quantum Machine
        if isinstance(qubits, int):
            self._qubits = Qureg(qubits)
        elif isinstance(qubits, Qureg):
            self._qubits = qubits
        else:
            raise TypeError("VirtualQuantumMachine.qubits", "one of [int, Qureg]", f"{type(qubits)}")

        if qubit_fidelity is not None:
            self._qubits.set_fidelity(qubit_fidelity)
            self._qubit_fidelity = qubit_fidelity

        if t1_coherence_time is not None:
            self._qubits.set_t1_time(t1_coherence_time)
            self._t1_times = t1_coherence_time

        if t2_coherence_time is not None:
            self._qubits.set_t2_time(t2_coherence_time)
            self._t2_times = t2_coherence_time

        if coupling_strength is not None:
            self._qubits.set_coupling_strength(coupling_strength)
            self._coupling_strength = coupling_strength

        # Describe the layout of Quantum Machine
        if layout is not None:
            assert isinstance(layout, Layout), TypeError("VirtualQuantumMachine.layout", "Layout", f"{type(layout)}")

        self._layout = layout

        # Describe the gate set of Quantum Machine
        if instruction_set is not None:
            assert isinstance(instruction_set, InstructionSet), \
                TypeError("VirtualQuantumMachine.instruction_set", "InstructionSet", f"{type(instruction_set)}")

        self._instruction_set = instruction_set

        if gate_fidelity is not None:
            assert isinstance(gate_fidelity, dict), \
                TypeError("VirtualQuantumMachine.gate_fidelity", "List", f"{type(gate_fidelity)}")
            assert len(gate_fidelity.keys()) == self._instruction_set.size(), ValueError(
                "VirtualQuantumMachine.gate_fidelity",
                f"equal to {self._instruction_set.size()}",
                f"{len(gate_fidelity.keys())}"
            )

        self._gate_fidelity = gate_fidelity

        # Describe the noise of Quantum Machine
        if noise_model is not None:
            assert isinstance(noise_model, NoiseModel), \
                TypeError("VirtualQuantumMachine.noise_model", "NoiseModel", f"{type(noise_model)}")

        self._noise_model = noise_model
