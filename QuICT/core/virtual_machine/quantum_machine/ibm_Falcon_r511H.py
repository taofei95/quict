""" Generate The Virtual Quantum Machine Model for IBM's ibmq_perth. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 5
iset = InstructionSet(GateType.cx, [GateType.id, GateType.rz, GateType.sx, GateType.x])
layout = Layout(qubit_number)
layout.add_edge(0, 1, directional=False, error_rate=1.0)
layout.add_edge(1, 3, directional=False, error_rate=1.0)
layout.add_edge(1, 2, directional=False, error_rate=1.0)
layout.add_edge(3, 5, directional=False, error_rate=1.0)
layout.add_edge(4, 5, directional=False, error_rate=1.0)
layout.add_edge(5, 6, directional=False, error_rate=1.0)

qubit_fidelity = [
    (0.0235999999999999, 0.0286), (0.0248, 0.0244), (0.0213999999999999, 0.034), (0.0172, 0.0151999999999999),
    (0.0244, 0.0255999999999999), (0.0268, 0.0286), (0.00760000000000005, 0.0054)
]
gate_fidelity = [
    0.000189050853796395, 0.000322397870901852, 0.000221780954602079, 0.000204854512294976, 0.000329270743491048,
    0.000492962189796472, 0.000288643832183614
]
coupling_strength = [
    (0, 1, 0.00791456540569041), (1, 3, 0.016519359317290316), (1, 2, 0.006611751464476401),
    (1, 0, 0.00791456540569041), (2, 1, 0.006611751464476401), (3, 5, 0.008237670417111709),
    (3, 1, 0.016519359317290316), (4, 5, 0.01358518914900636), (5, 6, 0.019817433730546313),
    (5, 4, 0.01358518914900636), (5, 3, 0.008237670417111709), (6, 5, 0.019817433730546313)
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
