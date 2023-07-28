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
