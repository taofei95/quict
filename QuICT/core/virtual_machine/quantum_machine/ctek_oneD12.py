""" Generate The Virtual Quantum Machine Model for Quantum Computing's CTEK OneD12. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 12
iset = InstructionSet(GateType.cz, [GateType.x, GateType.y, GateType.rx, GateType.ry])  # (2 / pi) or (-2 / pi) params
layout = Layout.linear_layout(qubit_number)
qubit_fidelity = [
    (0.9342, 0.8049), (0.9293, 0.8756), (0.9698, 0.9099), (0.9335, 0.8896), (0.9875, 0.8874), (0.9217, 0.8954),
    (0.9613, 0.9127), (0.9084, 0.8572), (0.9627, 0.9011), (0.8325, 0.7619), (0.9042, 0.8796), (0.7896, 0.7063)
]
gate_fidelity = [0.9973, 0.9975, 0.9972, 0.9946, 0.9963, 0.9972, 0.9969, 0.9982, 0.999, 0.9992, 0.9997, 0.9988]
coupling_strength = [
    (0, 1, 0.9885), (1, 2, 0.9756), (2, 3, 0.9786), (3, 4, 0.9857), (4, 5, 0.9614),
    (5, 6, 0.9756), (6, 7, 0.9401), (7, 8, 0.9717), (8, 9, 0.9354), (9, 10, 0.9641), (10, 11, 0.9522)
]
readout_frequency = [4.9650, 4.4430, 4.9160, 4.3520, 5.0960, 4.1970, 5.0270, 4.2880, 5.1880, 4.3750, 4.8820, 4.0140]
T1_times = [35.2, 46.9, 37.2, 35, 42, 35.1, 30.7, 19.2, 46.8, 31.1, 36.5, 49.5]
T2_times = [4, 3.1, 3.7, 2.9, 10.5, 2.2, 8.6, 2.6, 9.6, 2, 2.3, 2.5]

CTEKOneD12 = VirtualQuantumMachine(
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
