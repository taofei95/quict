""" Generate The Virtual Quantum Machine Model for Baidu's CloudIoPCAS. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 10
iset = InstructionSet(GateType.cz, [GateType.u3])   # TODO: change late, not find baidu's instruction set currently
layout = Layout.linear_layout(qubit_number)
qubit_fidelity = [
    (0.989, 0.927), (0.985, 0.909), (0.988, 0.923), (0.956, 0.915), (0.974, 0.831),
    (0.968, 0.871), (0.980, 0.907), (0.964, 0.919), (0.985, 0.936), (0.968, 0.912)
    ]
gate_fidelity = [0.9845, 0.9968, 0.9921, 0.9903, 0.9969, 0.9914, 0.9807, 0.9981, 0.9994, 0.9990]
coupling_strength = [
    (0, 1, 0.9450), (1, 2, 0.9450), (2, 3, 0.9230), (3, 4, 0.9419), (4, 5, 0.9370),
    (5, 6, 0.9560), (6, 7, 0.9419), (7, 8, 0.9240), (8, 9, 0.9430)
    ]
work_frequency = [5456, 4424, 5606, 4327, 5473, 4412, 5392, 4319, 5490, 4442]
readout_frequency = [6.663, 6.646, 6.627, 6.608, 6.593, 6.570, 6.554, 6.531, 6.510, 6.490]
T1_times = [43.7, 52.9, 21.6, 39.7, 14.8, 32.0, 28.8, 26.0, 24.1, 24.7]
T2_times = [11.1, 2.2, 4.3, 3.7, 4.6, 1.4, 5.2, 1.4, 3.0, 1.3]

BaiduCloudIoPCAS = VirtualQuantumMachine(
    qubits=qubit_number,
    instruction_set=iset,
    qubit_fidelity=qubit_fidelity,
    gate_fidelity=gate_fidelity,
    coupling_strength=coupling_strength,
    layout=layout,
    work_frequency=work_frequency,
    readout_frequency=readout_frequency,
    t1_coherence_time=T1_times,
    t2_coherence_time=T2_times
)
