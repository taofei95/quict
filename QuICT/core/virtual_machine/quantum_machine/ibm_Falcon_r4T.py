""" Generate The Virtual Quantum Machine Model for IBM's ibmq_quito. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 5
iset = InstructionSet(GateType.cx, [GateType.id, GateType.rz, GateType.sx, GateType.x])
layout = Layout(qubit_number)
layout.add_edge(0, 1, directional=False)
layout.add_edge(1, 3, directional=False)
layout.add_edge(1, 2, directional=False)
layout.add_edge(3, 4, directional=False)

qubit_fidelity = [(0.057, 0.0238), (0.0508, 0.022), (0.0744, 0.0548), (0.0526, 0.0158), (0.0568, 0.0224)]
gate_fidelity = [0.000287732, 0.000283475, 0.000249006, 0.00027076, 0.000550901]
coupling_strength = [
    (0, 1, 0.006980828540715911), (1, 3, 0.007490562309400067), (1, 2, 0.0090892457240202), (1, 0, 0.006980828540715911),
    (2, 1, 0.0090892457240202), (3, 4, 0.0145518931133225), (3, 1, 0.007490562309400067), (4, 3, 0.0145518931133225)
    ]
readout_frequency = [5.300688786, 5.080578878, 5.322183783, 5.163615471, 5.05233292]
T1_times = [89.20788692, 97.22909445, 95.50221618, 100.6880296, 106.0690951]
T2_times = [116.1136452, 115.737077, 101.2996922, 20.97791725, 148.1471507]

IBMFalconr4T = VirtualQuantumMachine(
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