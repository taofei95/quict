""" Generate The Virtual Quantum Machine Model for IBM's ibmq_manila. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 5
iset = InstructionSet(GateType.cx, [GateType.id, GateType.rz, GateType.sx, GateType.x])
layout = Layout.linear_layout(qubit_number)
qubit_fidelity = [(0.0394, 0.0148), (0.0206, 0.0136), (0.0368, 0.0084), (0.0276, 0.0148), (0.0464, 0.0074)]
gate_fidelity = [0.000221335, 0.00020993, 0.00034836, 0.000201412, 0.00059609]
coupling_strength = [
    (0, 1, 0.005987822496360373), (1, 2, 0.011587638206098094), (1, 0, 0.005987822496360373),
    (2, 1, 0.011587638206098094), (2, 3, 0.007210543234091954), (3, 4, 0.005606917264172867),
    (3, 2, 0.007210543234091954), (4, 3, 0.005606917264172867)
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
