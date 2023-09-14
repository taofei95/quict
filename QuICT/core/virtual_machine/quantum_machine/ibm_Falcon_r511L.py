""" Generate The Virtual Quantum Machine Model for IBM's ibmq_manila. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 5
iset = InstructionSet(GateType.cx, [GateType.id, GateType.rz, GateType.sx, GateType.x])
layout = Layout.linear_layout(qubit_number)
qubit_fidelity = [
    (0.9606, 0.9852), (0.9794, 0.9864), (0.9632000000000001, 0.9916), (0.9724, 0.9852), (0.9536, 0.9926)
]
gate_fidelity = [0.999778665, 0.99979007, 0.99965164, 0.999798588, 0.99940391]
coupling_strength = [
    (0, 1, 0.9940121775036397), (1, 2, 0.9884123617939019), (1, 0, 0.9940121775036397),
    (2, 1, 0.9884123617939019), (2, 3, 0.992789456765908), (3, 4, 0.9943930827358272),
    (3, 2, 0.992789456765908), (4, 3, 0.9943930827358272)
]
readout_frequency = [4.962289267, 4.837876927, 5.037242615, 4.950950146, 5.06512984]
T1_times = [167.1787673, 248.5348428, 126.8575858, 214.2549748, 134.0741238]
T2_times = [79.6549035, 74.22098881, 24.03598606, 76.07873032, 39.51155238]

IBMFalconr511L = VirtualQuantumMachine(
    qubits=qubit_number,
    instruction_set=iset,
    qubit_fidelity=qubit_fidelity,
    gate_fidelity=gate_fidelity,
    coupling_strength=coupling_strength,
    layout=layout,
    readout_frequency=readout_frequency,
    t1_coherence_time=T1_times,
    t2_coherence_time=T2_times
)
