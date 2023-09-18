""" Generate The Virtual Quantum Machine Model for IBM's ibmq_guadalupe. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 16
iset = InstructionSet(GateType.cx, [GateType.id, GateType.rz, GateType.sx, GateType.x])
layout = Layout(qubit_number)
layout.add_edge(0, 1, directional=False)
layout.add_edge(1, 4, directional=False)
layout.add_edge(1, 2, directional=False)
layout.add_edge(2, 3, directional=False)
layout.add_edge(3, 5, directional=False)
layout.add_edge(4, 7, directional=False)
layout.add_edge(5, 8, directional=False)
layout.add_edge(6, 7, directional=False)
layout.add_edge(7, 10, directional=False)
layout.add_edge(8, 11, directional=False)
layout.add_edge(8, 9, directional=False)
layout.add_edge(10, 12, directional=False)
layout.add_edge(11, 14, directional=False)
layout.add_edge(12, 15, directional=False)
layout.add_edge(12, 13, directional=False)
layout.add_edge(13, 14, directional=False)

qubit_fidelity = [
    (0.982, 0.9922), (0.9784, 0.9936), (0.9616, 0.984), (0.9782, 0.9928), (0.9724, 0.99), (0.9612, 0.9916),
    (0.9796000000000001, 0.9974), (0.9666000000000001, 0.9828), (0.9628, 0.989), (0.9452, 0.9914),
    (0.9704000000000002, 0.9946), (0.9528, 0.9936), (0.975, 0.9908), (0.9734, 0.9976), (0.9608, 0.9922),
    (0.9596000000000001, 0.9888)
]
gate_fidelity = [
    0.9998023728782298, 0.9996473040116634, 0.9995923733903923, 0.9997894768180752, 0.9997944883783644,
    0.9805050158300854, 0.9997784655391705, 0.999791485323481, 0.9997226031419031, 0.9996108421125682,
    0.9996059448125897, 0.9995530400548915, 0.9997561698014835, 0.9979208093206808, 0.9996886049009187,
    0.999718900009425
]
coupling_strength = [
    (0, 1, 0.9892706985602616), (1, 4, 0.993464142274743), (1, 2, 0.9912309518470119),
    (2, 3, 0.98713147578134), (3, 5, 0.9871052967593177), (4, 7, 0.9868666344679782),
    (5, 8, 0.9940839033910808), (6, 7, 0.9932413663593556), (7, 10, 0.9940969215338274),
    (8, 11, 0.9921844503702703), (8, 9, 0.9906079258353474), (10, 12, 0.9818841072810425),
    (11, 14, 0.9894388171959565), (12, 15, 0.9929504926450456), (12, 13, 0.9927326127792565),
    (13, 14, 0.9896837851605564)
]
readout_frequency = [
    5.11353166937853, 5.16074768310643, 5.31051908557267, 5.46953239499715, 5.35342862747041, 5.30322638119672,
    5.13648798168807, 5.20279516264014, 5.17454411984919, 5.25372616563932, 5.42679576525032, 5.38983849928948,
    5.2637173767519, 5.03786385264823, 5.20499169587972, 5.12630165266159
]
T1_times = [
    88.1134197154687, 126.978097231844, 79.1121982496341, 66.5550976104867, 100.262497675285, 85.3496053003645,
    100.158492545956, 105.314611787621, 110.892816202928, 127.094355345812, 126.76887415414, 56.7754880319254,
    106.262565611166, 124.594226997778, 97.8816665152367, 120.993332104466
]
T2_times = [
    95.5144674152651, 142.216783747765, 108.136767612862, 73.168878928193, 130.453774769872, 87.2863146063023,
    15.5469482116998, 128.577029661559, 130.993395129241, 111.012012115387, 120.096044587385, 59.2837097203424,
    104.740342708322, 91.2842748214204, 93.5770481327628, 177.888443283363
]

IBMFalconr4p = VirtualQuantumMachine(
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
