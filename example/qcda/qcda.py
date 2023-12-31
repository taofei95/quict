import random

import os
from scipy.stats import unitary_group

from QuICT.core import Circuit, Layout
from QuICT.core.gate import GateType
from QuICT.qcda.synthesis.gate_transform import *
from QuICT.qcda.qcda import QCDA


typelist = [GateType.rx, GateType.ry, GateType.rz, GateType.x, GateType.y, GateType.z, GateType.cx]


if __name__ == '__main__':
    layout_path = os.path.join(os.path.dirname(__file__), "../layout/ibmqx2_layout.json")
    layout = Layout.load_file(layout_path)

    circuit = Circuit(5)
    circuit.random_append(typelist=typelist)
    target = random.sample(range(5), 3)
    CCRz(np.pi / 3) | circuit(target)
    circuit.random_append(typelist=typelist)
    target = random.sample(range(5), 3)
    CSwap | circuit(target)
    circuit.random_append(typelist=typelist)
    matrix = unitary_group.rvs(2 ** 3)
    target = random.sample(range(5), 3)
    Unitary(matrix) | circuit(target)
    circuit.random_append(typelist=typelist)
    circuit.draw(filename="before_qcda")

    qcda = QCDA()
    qcda.add_gate_transform(USTCSet)
    qcda.add_default_optimization()
    qcda.add_mapping(layout)
    qcda.add_gate_transform(USTCSet)
    circuit_phy = qcda.compile(circuit)
    circuit_phy.draw(filename="after_qcda")
