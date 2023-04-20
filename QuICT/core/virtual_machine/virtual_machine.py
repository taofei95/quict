from typing import Union, List

from QuICT.core import Qureg, Layout, Circuit
from QuICT.core.noise import NoiseModel
from QuICT.qcda.synthesis import InstructionSet
from QuICT.tools.exception.core import TypeError, ValueError


class VirtualQuantumMachine:
    """ The Class store the information about Quantum Machine. """
    @property
    def qubit_number(self) -> int:
        """ Return the number of qubits. """
        return len(self._qubits)

    @property
    def qubits(self) -> Qureg:
        """ Return the Qureg of current Machine. """
        return self._qubits

    @qubits.setter
    def qubits(self, qubits: Qureg):
        assert isinstance(qubits, Qureg)
        self._qubits = qubits

    @property
    def instruction_set(self) -> InstructionSet:
        """ Return the instruction set of current Machine. """
        return self._instruction_set

    @instruction_set.setter
    def instruction_set(self, ins: InstructionSet):
        assert isinstance(ins, InstructionSet), \
            TypeError("VirtualQuantumMachine.instruction_set", "InstructionSet", f"{type(ins)}")
        self._instruction_set = ins

    @property
    def layout(self) -> Layout:
        """ Return the layout of current Machine. """
        return self._layout

    @layout.setter
    def layout(self, layout: Layout):
        assert isinstance(layout, Layout), TypeError("VirtualQuantumMachine.layout", "Layout", f"{type(layout)}")
        assert layout.qubit_number == self.qubit_number
        self._layout = layout

    @property
    def qubit_fidelity(self) -> list:
        """ Return the fidelity of each qubits. """
        return self._qubit_fidelity

    @qubit_fidelity.setter
    def qubit_fidelity(self, qf: list):
        self._qubits.set_fidelity(qf)
        self._qubit_fidelity = qf

    @property
    def t1_time(self) -> list:
        """ Return the t1 coherence strength of each qubits. """
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
    def coupling_strength(self) -> list:
        return self._coupling_strength

    @coupling_strength.setter
    def coupling_strength(self, cs: list):
        self._qubits.set_coupling_strength(cs)
        self._coupling_strength = cs

    @property
    def gate_fidelity(self) -> dict:
        return self._gate_fidelity

    @gate_fidelity.setter
    def gate_fidelity(self, gf: dict):
        assert isinstance(gf, dict), TypeError("VirtualQuantumMachine.gate_fidelity", "List", f"{type(gf)}")
        assert len(gf.keys()) == self._instruction_set.size() and self._gate_in_set(gf.keys()), \
            ValueError(
                "VirtualQuantumMachine.gate_fidelity", f"equal to {self._instruction_set.size()}", f"{len(gf.keys())}"
            )

        self._gate_fidelity = gf

    @property
    def noise_model(self) -> NoiseModel:
        return self._noise_model

    @noise_model.setter
    def noise_model(self, nm: NoiseModel):
        assert isinstance(nm, NoiseModel), TypeError("VirtualQuantumMachine.noise_model", "NoiseModel", f"{type(nm)}")
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

        self._qubit_fidelity = None
        if qubit_fidelity is not None:
            self.qubit_fidelity = qubit_fidelity

        self._t1_times = None
        if t1_coherence_time is not None:
            self.t1_times = t1_coherence_time

        self._t2_times = None
        if t2_coherence_time is not None:
            self.t2_times = t2_coherence_time

        self._coupling_strength = None
        if coupling_strength is not None:
            self.coupling_strength = coupling_strength

        # Describe the layout of Quantum Machine
        self._layout = None
        if layout is not None:
            self.layout = layout

        # Describe the gate set of Quantum Machine
        self._instruction_set = None
        if instruction_set is not None:
            self.instruction_set = instruction_set

        self._gate_fidelity = None
        if gate_fidelity is not None:
            self.gate_fidelity = gate_fidelity

        # Describe the noise of Quantum Machine
        self._noise_model = None
        if noise_model is not None:
            self.noise_model = noise_model

    def _gate_in_set(self, gates: list) -> bool:
        if self._instruction_set is None:
            return False

        current_gateset = self._instruction_set.gates
        for gatetype in gates:
            if gatetype not in current_gateset:
                return False

        return True

    #################    Quantum Circuit Auto Design    ##################
    def evaluate(self, circuit: Circuit) -> float:
        """ Return the fidelity of Circuit. """
        pass

    def transpile(self, circuit: Circuit) -> Circuit:
        """ Return the circuit that can run on this Quantum Machine. 
        Consider the layout and instruction set of current Quantum Machine.
        """
        pass

    #################    Quantum Circuit Auto Design    ##################
    def get_benchmark(self) -> List[Circuit]:
        """ Get the benchmark circuit for this Quantum Machine. """
        pass
