from typing import Union, List, Dict

from QuICT.core import Qureg, Layout
from QuICT.tools.exception.core import TypeError

from .instruction_set import InstructionSet


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
    def preparation_fidelity(self) -> list:
        """ Return the fidelity of each qubits. """
        return self._qsp_fidelity

    @preparation_fidelity.setter
    def preparation_fidelity(self, qsp: list):
        self._qubits.set_preparation_fidelity(qsp)
        self._qsp_fidelity = qsp

    @property
    def t1_times(self) -> list:
        """ Return the t1 coherence strength of each qubits. """
        return self._t1_times

    @t1_times.setter
    def t1_times(self, t1: list):
        self._qubits.set_t1_time(t1)
        self._t1_times = t1

    @property
    def t2_times(self) -> list:
        return self._t2_times

    @t2_times.setter
    def t2_times(self, t2: list):
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
    def gate_fidelity(self, gf):
        self._instruction_set.register_one_qubit_fidelity(gf)
        self._gate_fidelity = gf

    def __init__(
        self,
        qubits: Union[int, Qureg],
        instruction_set: InstructionSet,
        name: str = None,
        qubit_fidelity: List[float] = None,
        preparation_fidelity: List[float] = None,
        gate_fidelity: Union[float, List, Dict] = None,
        t1_coherence_time: List[float] = None,
        t2_coherence_time: List[float] = None,
        coupling_strength: List[tuple] = None,
        layout: Layout = None
    ):
        """
        Args:
            qubits (Union[int, Qureg]): The qubit number or the Qureg which is the list of Qubit.
            instruction_set (InstructionSet): The set of quantum gates which Quantum Machine supports.
            name (str): The name of quantum machine.
            qubit_fidelity (list, optional): The readout fidelity for each qubit. Defaults to None.
            preparation_fidelity (list, optional): The state preparation fidelity for each qubit. Defaults to None.
            gate_fidelity (Union[float, dict], optional): The fidelity for single qubit quantum gate. Defaults to None.
            t1_coherence_time (list, optional): The t1 coherence time for each qubit. Defaults to None.
            t2_coherence_time (list, optional): The t2 coherence time for each qubit. Defaults to None.
            coupling_strength (list, optional): The coupling strength between the qubits. Defaults to None.
            layout (Layout, optional): The description of physical topology of Quantum Machine. Defaults to None.

        Raises:
            TypeError: The wrong type about input.
        """
        self.name = name if name is not None else "Quantum_Machine"

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

        self._qsp_fidelity = None
        if preparation_fidelity is not None:
            self.preparation_fidelity = preparation_fidelity

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
