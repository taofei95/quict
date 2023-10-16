from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.utility.circuit_cost import CircuitCost
from QuICT.core.virtual_machine.quantum_machine import OriginalKFC6130


def test_fidelity_circuit_cost():
    n_qubit = 4
    n_gate = 10
    circ = Circuit(n_qubit)
    circ.random_append(n_gate, typelist=[GateType.cz, GateType.u3])
    cost = CircuitCost(backend=OriginalKFC6130)
    print(cost.evaluate_cost(circ))


def test_composite_gate():
    cg1 = CompositeGate(gates=[H & 0, H & 1, CX & [0, 1]])
    cg2 = CompositeGate(gates=[CZ & [0, 1], Rz(1) & 0, Rx(2) & 2, cg1])
    circ = Circuit(3)
    circ.extend(cg1)
    circ.extend(cg2)

    cost = CircuitCost()
    c1 = cost.evaluate_cost(circ)
    circ.flatten_gates()
    c2 = cost.evaluate_cost(circ)
    assert c1 == c2
