""" Generate The Virtual Quantum Machine Model for IBM's ibmq_perth. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 7
iset = InstructionSet(GateType.cx, [GateType.id, GateType.rz, GateType.sx, GateType.x])
layout = Layout(qubit_number)
layout.add_edge(0, 1, directional=False, error_rate=1.0)
layout.add_edge(1, 3, directional=False, error_rate=1.0)
layout.add_edge(1, 2, directional=False, error_rate=1.0)
layout.add_edge(3, 5, directional=False, error_rate=1.0)
layout.add_edge(4, 5, directional=False, error_rate=1.0)
layout.add_edge(5, 6, directional=False, error_rate=1.0)

qubit_fidelity = [
    (0.9764000000000002, 0.9714), (0.9752, 0.9756), (0.9786000000000001, 0.966), (0.9828, 0.9848000000000001),
    (0.9756, 0.9744000000000002), (0.9732, 0.9714), (0.9924, 0.9946)
]
gate_fidelity = [
    0.9998109491462036, 0.9996776021290982, 0.9997782190453979, 0.999795145487705, 0.9996707292565089,
    0.9995070378102036, 0.9997113561678164
]
coupling_strength = [
    (0, 1, 0.9920854345943095), (1, 3, 0.9834806406827097), (1, 2, 0.9933882485355237),
    (1, 0, 0.9920854345943095), (2, 1, 0.9933882485355237), (3, 5, 0.9917623295828883),
    (3, 1, 0.9834806406827097), (4, 5, 0.9864148108509936), (5, 6, 0.9801825662694537),
    (5, 4, 0.9864148108509936), (5, 3, 0.9917623295828883), (6, 5, 0.9801825662694537)
]
readout_frequency = [
    5.15753760509277, 5.03368042862207, 4.86265292157984, 5.12510747322561, 5.15921857335882, 4.97861306769581,
    5.15664106594405]
T1_times = [
    154.204089223613, 142.137880585714, 256.43118479513, 249.357307156306, 153.953218426146, 181.4164137808,
    152.618773168032
]
T2_times = [
    99.7277464629721, 51.5508910960636, 95.0987664893622, 175.56808867738, 141.323874701537, 140.701524858115,
    103.557534745468
]

IBMFalconr511H = VirtualQuantumMachine(
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
