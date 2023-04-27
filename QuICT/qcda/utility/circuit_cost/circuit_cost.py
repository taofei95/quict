from math import log

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.virtual_machine.virtual_machine import VirtualQuantumMachine

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

    def __init__(self, backend: VirtualQuantumMachine = None):
        # qubit_info[i]: Dict {
        #     'T1': float,
        #     'T2': float,
        #     'prob_meas0_prep1': float,
        #     'prob_meas1_prep0': float,
        # }
        self.qubit_info = {}
        # gate_fidelity[qubit tuple]: Dict {
        #     GateType: fidelity
        # }
        self.gate_fidelity = {}

        self.backend = backend
        self._load_backend_model()

    def _load_backend_model(self):
        pass

    def gate_cost(self, gate, a=1., c=1.):
        tot_time = 0
        qubits = tuple(gate.cargs + gate.targs)
        for q in qubits:
            tot_time += self.qubit_info[q]['T1'] + self.qubit_info[q]['T2']
        avg_time = tot_time / (gate.controls + gate.targets)
        gate_f = self.gate_fidelity[qubits][gate.type]
        # print(-100 * log(gate_f) + 1)
        # return (-100 * log(gate_f) + 1) / avg_time / 1.2
        return (-log(gate_f) * a + 1) / avg_time / c

    def evaluate(self, circuit: Circuit, a=400, c=2.3):
        cost = 0
        for g in circuit.gates:
            cost += self.gate_cost(g, a=a, c=c)
        return cost
