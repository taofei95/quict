from functools import reduce
from math import log

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.virtual_machine import InstructionSet
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
        """
        Args:
            backend(VirtualQuantumMachine): Backend machine.
        """
        self.backend = backend

        n_qubit = self.backend.qubit_number
        self.max_T1 = max(self.backend.t1_times) if self.backend.t1_times else 0
        self.max_T2 = max(self.backend.t2_times) if self.backend.t2_times else 0

        # qubit_info[i]: Dict {
        #     'T1': float,
        #     'T2': float,
        #     'prob_meas0_prep1': float,
        #     'prob_meas1_prep0': float,
        # }
        # self.qubit_info = {}
        # gate_fidelity[qubit tuple]: Dict {
        #     GateType: fidelity
        # }
        # self.gate_fidelity = {}

    @staticmethod
    def from_quest_data(model_info):
        """
        Create a CircuitCost object from QuEST data.

        Wang, Hanrui, et al. "QuEst: Graph Transformer for Quantum Circuit Reliability Estimation."
        arXiv preprint arXiv:2210.16724 (2022).
        """

        qubit_info = model_info['qubit']
        gate_info = model_info['gate']
        one_qubit_gates = [
            getattr(GateType, gate_str) for gate_str in
            reduce(set.__or__, [set(val.keys()) for key, val in gate_info.items() if len(key) == 1])
        ]
        two_qubit_str = list(reduce(
            set.__or__,
            [set(val.keys()) for key, val in gate_info.items() if len(key) > 1]
        ))[0]

        two_qubit_gate = getattr(GateType, two_qubit_str)
        backend = VirtualQuantumMachine(
            len(qubit_info),
            InstructionSet(two_qubit_gate, one_qubit_gates)
        )

        backend.t1_times = [0] * len(qubit_info)
        for q in qubit_info:
            if 'T1' in qubit_info[q]:
                backend.t1_times[q] = qubit_info[q]['T1']

        backend.t2_times = [0] * len(qubit_info)
        for q in qubit_info:
            if 'T2' in qubit_info[q]:
                backend.t2_times[q] = qubit_info[q]['T2']

        avg_t1 = sum(backend.t1_times) / len(list(filter(lambda x: x, backend.t1_times)))
        avg_t2 = sum(backend.t2_times) / len(list(filter(lambda x: x, backend.t2_times)))
        for q in qubit_info:
            if backend.t1_times[q] == 0:
                backend.t2_times[q] = avg_t1
            if backend.t2_times[q] == 0:
                backend.t2_times[q] = avg_t2

        backend.gate_fidelity = [{}] * len(qubit_info)
        backend.coupling_strength = {}
        for qubits in gate_info:
            if len(qubits) == 1:
                for g, f in gate_info[qubits].items():
                    backend.gate_fidelity[qubits[0]][getattr(GateType, g)] = 1 - f
            else:
                backend.coupling_strength[qubits] = 1 - gate_info[qubits][two_qubit_str]

        backend.qubit_fidelity = [1] * len(qubit_info)
        for q in qubit_info:
            avg_fidelity = 1 - (qubit_info[q]['prob_meas0_prep1'] + qubit_info[q]['prob_meas1_prep0']) / 2
            backend.qubit_fidelity[q] = avg_fidelity

        return CircuitCost(backend)

    def _gate_fidelity(self, gate: BasicGate):
        qubits = tuple(gate.cargs + gate.targs)
        if gate.type not in self.backend.instruction_set.one_qubit_gates and \
                gate.type != self.backend.instruction_set.two_qubit_gate:
            assert False, f'Gate type {gate.type} not in instruction set'
        if gate.controls + gate.targets == 1:
            gate_f = self.backend.gate_fidelity[qubits[0]][gate.type] if \
                self.backend.gate_fidelity else 1
        else:
            if self.backend.layout and not self.backend.layout.check_edge(*qubits):
                assert False, f'Qubits {qubits} not connected'

            gate_f = 1
            if self.backend.coupling_strength:
                if qubits in self.backend.coupling_strength:
                    gate_f = self.backend.coupling_strength[qubits]
                elif tuple(reversed(qubits)) in self.backend.coupling_strength:
                    gate_f = self.backend.coupling_strength[tuple(reversed(qubits))]

        return gate_f

    def gate_cost(self, gate: BasicGate, fidelity_coef=1):
        """
        Calculate cost of a gate.

        Args:
            gate(BasicGate): Gate to calculate cost.
            fidelity_coef(float): Coefficient of fidelity in cost function.

        Returns:
            float: Cost of the gate.
        """
        tot_time = 0
        qubits = tuple(gate.cargs + gate.targs)
        for q in qubits:
            if self.backend.t1_times:
                tot_time += self.backend.t1_times[q]
            if self.backend.t2_times:
                tot_time += self.backend.t2_times[q]

        avg_time = tot_time / (gate.controls + gate.targets) / (self.max_T1 + self.max_T2) if \
            self.max_T1 + self.max_T2 else 1

        gate_f = self._gate_fidelity(gate)
        return (-fidelity_coef * log(gate_f) + 1) / avg_time

    def evaluate(self, circuit: Circuit):
        """
        Estimate the fidelity of a circuit.

        Args:
            circuit(Circuit): Circuit to evaluate.

        Returns:
            float: Estimated fidelity of the circuit.
        """
        qubit_f = [1] * circuit.width()
        qubit_gate_count = [0] * circuit.width()

        for g in circuit.gates:
            gate_f = self._gate_fidelity(g)
            for q in g.cargs + g.targs:
                qubit_f[q] *= gate_f
                qubit_gate_count[q] += 1

        for q in range(circuit.width()):
            tot_time, cnt = 0, 0
            if self.backend.t1_times:
                tot_time += self.backend.t1_times[q]
                cnt += 1
            if self.backend.t2_times:
                tot_time += self.backend.t2_times[q]
                cnt += 1
            avg_time = tot_time / cnt

            if qubit_gate_count[q]:
                qubit_f[q] *= np.exp(-1 * qubit_gate_count[q] / avg_time)
                if self.backend.qubit_fidelity:
                    qubit_f[q] *= self.backend.qubit_fidelity[q]
        circ_f = np.prod(qubit_f)

        return circ_f

    def evaluate_backup(self, circuit: Circuit, fidelity_coef=1):
        """
        Evaluate cost of a circuit.

        Args:
            circuit(Circuit): Circuit to evaluate.
            fidelity_coef(float): Coefficient of fidelity in cost function.

        Returns:
            float: Cost of the circuit.
        """

        cost = 0
        for g in circuit.gates:
            cost += self.gate_cost(g, fidelity_coef)
        return cost
