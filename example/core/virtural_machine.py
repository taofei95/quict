import os

from QuICT.core import Layout
from QuICT.core.virtual_machine import InstructionSet, VirtualQuantumMachine
from QuICT.core.utils import GateType


def build_VQM():
    iset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])
    l_path = os.path.join(os.path.dirname(__file__), "../layout/grid_3x3.json")
    layout = Layout.load_file(l_path)
    cs = [
        (0, 1, 0.9), (0, 3, 0.9), (1, 2, 0.91), (1, 4, 0.91), (2, 5, 0.8), (3, 4, 0.6),
        (3, 6, 0.6), (4, 5, 0.5), (4, 7, 0.5), (5, 8, 0.45), (6, 7, 0.45), (7, 8, 0.45),
    ]
    vqm = VirtualQuantumMachine(
        qubits=9,
        instruction_set=iset,
        qubit_fidelity=[0.9] * 9,
        t1_coherence_time=[30.1] * 9,
        layout=layout,
        coupling_strength=cs,
        noise_model=None
    )
    vqm.t2_times = [2.3] * 9
    print(vqm.qubits[2])
    print(vqm.qubits.coupling_strength)
    print(vqm.layout)
    print(vqm.instruction_set.gates)


if __name__ == "__main__":
    build_VQM()