""" Generate The Virtual Quantum Machine Model for Quafu Quantum's ScQ-P10 Chip. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 10
instruction_set = InstructionSet(GateType.cx, [GateType.rx, GateType.rz, GateType.ry, GateType.h])
layout = Layout.linear_layout(qubit_number)
coupling_strength = [
    (0, 1, 0.9787), (1, 2, 0.9564), (2, 3, 0.949), (3, 4, 0.963), (4, 5, 0.9669),
    (5, 6, 0.9663), (6, 7, 0.956), (7, 8, 0.9741), (8, 9, 0.9909)
]
T1_times = [26.79, 29.38, 58.96, 18.92, 43.38, 47.37, 37.88, 19.5, 31.51, 23.43]
T2_times = [2.33, 1.82, 2.31, 1.61, 2.49, 2.14, 1.97, 1.33, 1.72, 3.56]
work_frequency = [5310, 4681, 5367, 4702, 5299, 4531, 5255, 4627, 5275, 4687]
readout_frequency = [6663, 6646, 6627, 6608, 6593, 6570, 6554, 6531, 6510, 6490]


QuafuScQP10 = VirtualQuantumMachine(
    qubits=qubit_number,
    instruction_set=instruction_set,
    coupling_strength=coupling_strength,
    layout=layout,
    work_frequency=work_frequency,
    readout_frequency=readout_frequency,
    t1_coherence_time=T1_times,
    t2_coherence_time=T2_times,
)
