import random

import os
import numpy as np
from scipy.stats import unitary_group

from QuICT.core import Circuit, Layout
from QuICT.core.gate import GateType, CCRz, CSwap, Unitary
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet
from QuICT.core.virtual_machine.special_set import USTCSet
from QuICT.qcda.qcda import QCDA


typelist = [GateType.rx, GateType.ry, GateType.rz, GateType.x, GateType.y, GateType.z, GateType.cx]
cli_tlist = [GateType.cx, GateType.x, GateType.y, GateType.z, GateType.h]


def qcda_workflow():
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


def auto_qcda_with_qm():
    layout_path = os.path.join(os.path.dirname(__file__), "../layout/ibmqx2_layout.json")
    layout = Layout.load_file(layout_path)
    iset = InstructionSet(GateType.cx, [GateType.rz, GateType.h])
    vqm = VirtualQuantumMachine(
        qubits=5, instruction_set=iset, layout=layout
    )

    circuit = Circuit(5)
    circuit.random_append(20, typelist=typelist, random_params=True)
    print("The original Quantum Circuit.")
    circuit.draw("command")

    qcda = QCDA()
    circuit_phy = qcda.auto_compile(circuit, vqm)
    print("The suitable Quantum Circuit for the given Quantum Machine.")
    circuit_phy.draw('command', flatten=True)


if __name__ == '__main__':
    auto_qcda_with_qm()
