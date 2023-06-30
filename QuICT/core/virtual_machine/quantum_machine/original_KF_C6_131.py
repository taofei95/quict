""" Generate The Virtual Quantum Machine Model for Origin Quantum's KF-C6-131 Chip. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 6
iset = InstructionSet(GateType.cz, [GateType.u3])
layout = Layout.linear_layout(qubit_number)
qubit_fidelity = [(0.971, 0.897), (0.922, 0.801), (0.978, 0.883), (0.960, 0.885), (0.968, 0.901), (0.944, 0.868)]
gate_fidelity = [0.9989, 0.9989, 0.9961, 0.9980, 0.9987, 0.9982]
coupling_strength = [(0, 1, 0.9851), (1, 2, 0.9619), (2, 3, 0.7014), (3, 4, 0.8256), (4, 5, 0.7132)]
work_frequency = [5429.197, 4995.75, 5671.73, 5111, 5850.218, 4963.037]
T1_times = [12, 22, 4, 14, 9, 16]
T2_times = [2.7, 9.3, 7.1, 17.8, 10.3, 3.3]

OriginalKFC6131 = VirtualQuantumMachine(
    qubits=qubit_number,
    instruction_set=iset,
    qubit_fidelity=qubit_fidelity,
    gate_fidelity=gate_fidelity,
    coupling_strength=coupling_strength,
    layout=layout,
    work_frequency=work_frequency,
    t1_coherence_time=T1_times,
    t2_coherence_time=T2_times
)
