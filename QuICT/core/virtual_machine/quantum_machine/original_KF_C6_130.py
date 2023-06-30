""" Generate The Virtual Quantum Machine Model for Origin Quantum's KF-C6-130 Chip. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 6
iset = InstructionSet(GateType.cz, [GateType.u3])
layout = Layout.linear_layout(qubit_number)
qubit_fidelity = [(0.989, 0.965), (0.950, 0.859), (0.975, 0.951), (0.958, 0.923), (0.984, 0.967), (0.914, 0.845)]
gate_fidelity = [0.9993, 0.9990, 0.9990, 0.9991, 0.9992, 0.9992]
coupling_strength = [(0, 1, 0.9909), (1, 2, 0.9881), (2, 3, 0.9707), (3, 4, 0.9834), (4, 5, 0.9858)]
work_frequency = [5442, 4470, 5319, 4696, 5214.995, 4579.685]
T1_times = [17, 30, 20, 32, 36, 28]
T2_times = [12.6, 2.3, 2.6, 6.6, 3.3, 5.4]

OriginalKFC6130 = VirtualQuantumMachine(
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
