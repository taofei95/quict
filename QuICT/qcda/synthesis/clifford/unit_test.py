import pytest

import numpy as np

from QuICT.core.gate import build_gate, CompositeGate, GateType
from QuICT.qcda.synthesis.clifford.disentangler import PauliOperator

def test_conjugate_action():
    pauli_list = [GateType.id, GateType.x, GateType.y, GateType.z]
    clifford_single = [GateType.h, GateType.s, GateType.x, GateType.y, GateType.z]

    # test clifford_single
    for pauli in pauli_list:
        for clifford in clifford_single:
            p = PauliOperator([pauli])
            clifford_gate = build_gate(clifford, 0)
            p.conjugate_act(clifford_gate)

            pauli_gate = build_gate(pauli, 0)
            gates = CompositeGate()
            if clifford != GateType.s:
                with gates:
                    clifford_gate & 0
                    pauli_gate & 0
                    clifford_gate & 0
            else:
                clifford_inverse_gate = build_gate(GateType.sdg, 0)
                with gates:
                    clifford_inverse_gate & 0
                    pauli_gate & 0
                    clifford_gate & 0
            assert np.allclose(p.gates.matrix() * p.phase, gates.matrix())
    
    # test cx
    for pauli_0 in pauli_list:
        for pauli_1 in pauli_list:
            p = PauliOperator([pauli_0, pauli_1])
            cx_gate = build_gate(GateType.cx, [0, 1])
            p.conjugate_act(cx_gate)

            pauli_gate_0 = build_gate(pauli_0, 0)
            pauli_gate_1 = build_gate(pauli_1, 1)
            gates = CompositeGate()
            with gates:
                cx_gate & [0, 1]
                pauli_gate_0 & 0
                pauli_gate_1 & 1
                cx_gate & [0, 1]
            assert np.allclose(p.gates.matrix() * p.phase, gates.matrix())


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
