from math import log

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from QuICT.core.utils import GateType


class StaticCircuitCost(object):
    """
    Static cost of quantum circuit.
    """

    NISQ_GATE_COST = {
        GateType.id: 0, GateType.x: 1, GateType.y: 1, GateType.z: 1, GateType.h: 1,
        GateType.t: 1, GateType.tdg: 1, GateType.s: 1, GateType.sdg: 1, GateType.u1: 1,
        GateType.u2: 2, GateType.u3: 2, GateType.rx: 1, GateType.ry: 1, GateType.rz: 1,
        GateType.cx: 2, GateType.cy: 4, GateType.cz: 4, GateType.ch: 8, GateType.swap: 6,
        GateType.cswap: 8, GateType.rxx: 9, GateType.ryy: 9, GateType.rzz: 5,
        GateType.cu3: 10, GateType.ccx: 21, GateType.measure: 9
    }

    def __init__(self, cost_dict=None):
        if cost_dict is None:
            self.cost_dict = self.NISQ_GATE_COST
        self.cost_dict = cost_dict

    def __getitem__(self, gate_type):
        """
        Get cost of a gate type. Subscript can be a GateType or string of a gate type.

        Args:
            gate_type(GateType/str): Gate type

        Returns:
            int: Cost of the gate type
        """
        if isinstance(gate_type, str):
            gate_type = GateType(gate_type)

        if gate_type in self.cost_dict:
            return self.cost_dict[gate_type]
        else:
            return 0

    def cost(self, circuit):
        """
        Calculate cost of a circuit based on static gate cost.
        """
        if isinstance(circuit, Iterable):
            return sum(self[n.gate.type] for n in circuit)
        else:
            return sum(self[n.gate.type] for n in circuit.gates)


class CircuitCost(object):
    """
    Measure of gate cost in quantum circuit.
    """

    def __init__(self, backend=None):
        self.backend = backend

    def _basic_cost(self, gate_type: GateType):
        # TODO compute based on backend
        return StaticCircuitCost.NISQ_GATE_COST[gate_type]

    def _relax_time_const(self):
        # TODO compute based on backend
        return 100

    def _mapping_distance(self, gate: BasicGate):
        # TODO decompose gate into backend.basic_gates
        # TODO calculate distance of each 2-qubit gate in backend.topological_structure
        return 0

    def _gate_cost(self, circ: Circuit, gate: BasicGate):
        tot_relax_t = sum([circ(q).t1 + circ(q).t2 for q in gate.cargs + gate.targs])
        avg_relax_t = tot_relax_t / (gate.controls + gate.targets)

        # TODO consider mapping distance
        return self._basic_cost(gate.type) / gate.fidelity * (self._relax_time_const() / avg_relax_t)

    def cost(self, circ: Circuit):
        cost = 0
        for gate in circ.gates:
            cost += self._gate_cost(circ, gate)
        return cost
