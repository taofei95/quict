""" Generate The Virtual Quantum Machine Model for Quafu Quantum's ScQ-P18 Chip. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 18
instruction_set = InstructionSet(GateType.cx, [GateType.rx, GateType.rz, GateType.ry, GateType.h])
layout = Layout.linear_layout(qubit_number)
coupling_strength = [
    (1, 0, 0.9828), (2, 1, 0.9775), (3, 2, 0.95), (4, 3, 0.95), (5, 4, 0.9821),
    (6, 5, 0.965), (6, 7, 0.97), (8, 7, 0.979), (9, 8, 0.9602), (10, 9, 0.95),
    (11, 10, 0.95), (12, 11, 0.95), (14, 13, 0.95), (15, 14, 0.95), (16, 15, 0.95),
    (17, 16, 0.95)
]
T1_times = [
    57.56, 27.89, 41.44, 40.1, 28.99, 36.9, 26.84, 19.76, 23.22, 27.31,
    41.15, 28.12, 45.12, 29.8, 45.12, 37.28, 40.84, 41.43
]
T2_times = [
    4.24, 3.7, 4.83, 2.3, 5.69, 2.05, 3.42, 2.44, 3.53, 1.32, 2.78, 1.49,
    2.95, 1.57, 5.47, 3.07, 5.15, 3.12
]
work_frequency = [
    4590, 5020, 4620, 5071, 4500, 4982, 4566, 5091, 4646, 5038, 4605, 4993,
    4545, 4869, 4578, 5048, 4682, 5125
]
readout_frequency = [
    6776, 6759, 6736, 6712, 6693, 6675, 6649, 6633, 6625, 6647, 6665, 6692,
    6705, 6725, 6751, 6770, 6787, 6807
]


QuafuScQP18 = VirtualQuantumMachine(
    qubits=qubit_number,
    instruction_set=instruction_set,
    coupling_strength=coupling_strength,
    layout=layout,
    work_frequency=work_frequency,
    readout_frequency=readout_frequency,
    t1_coherence_time=T1_times,
    t2_coherence_time=T2_times,
)
