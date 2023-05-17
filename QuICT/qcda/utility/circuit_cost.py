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

        t1_times = [0] * len(qubit_info)
        for q in qubit_info:
            if 'T1' in qubit_info[q]:
                t1_times[q] = qubit_info[q]['T1']

        t2_times = [0] * len(qubit_info)
        for q in qubit_info:
            if 'T2' in qubit_info[q]:
                t2_times[q] = qubit_info[q]['T2']

        gate_fidelity = [{}] * len(qubit_info)
        coupling_strength = []
        for qubits in gate_info:
            if len(qubits) == 1:
                for g, f in gate_info[qubits].items():
                    gate_fidelity[qubits[0]][getattr(GateType, g)] = 1 - f
            else:
                coupling_strength.append(qubits + (1 - gate_info[qubits][two_qubit_str], ))

        qubit_fidelity = [1] * len(qubit_info)
        for q in qubit_info:
            avg_fidelity = 1 - (qubit_info[q]['prob_meas0_prep1'] + qubit_info[q]['prob_meas1_prep0']) / 2
            qubit_fidelity[q] = avg_fidelity

        backend.t1_times = t1_times
        backend.t2_times = t2_times
        backend.coupling_strength = coupling_strength
        backend.qubit_fidelity = qubit_fidelity
        backend.gate_fidelity = gate_fidelity

        return CircuitCost(backend)

    def _gate_fidelity(self, gate: BasicGate):
        qubits = tuple(gate.cargs + gate.targs)
        if gate.type not in self.backend.instruction_set.one_qubit_gates and \
                gate.type != self.backend.instruction_set.two_qubit_gate:
            assert False, f'Gate type {gate.type} not in instruction set'
        if gate.controls + gate.targets == 1:
            if isinstance(self.backend.qubits[qubits[0]].gate_fidelity, dict):
                gate_f = self.backend.qubits[qubits[0]].gate_fidelity.get(gate.type, 1.)
            else:
                gate_f = self.backend.qubits[qubits[0]].gate_fidelity

        else:
            if self.backend.layout and not self.backend.layout.check_edge(*qubits):
                assert False, f'Qubits {qubits} not connected'

            gate_f = self.backend.qubits.coupling_strength[qubits[0]][qubits[1]]

        return gate_f

    def _qubit_avg_relax_time(self, q):
        tot_time, cnt = 0, 0
        if self.backend.qubits[q].T1:
            tot_time += self.backend.qubits[q].T1
            cnt += 1
        if self.backend.qubits[q].T2:
            tot_time += self.backend.qubits[q].T2
            cnt += 1
        return tot_time / cnt

    def evaluate_fidelity(self, circuit: Circuit):
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
            if qubit_gate_count[q]:
                # avg_time = self._qubit_avg_relax_time(q)
                # qubit_f[q] *= np.exp(-1 * qubit_gate_count[q] / avg_time)
                qubit_f[q] *= self.backend.qubits[q].fidelity
        circ_f = np.prod(qubit_f)

        return circ_f

    def evaluate(self, circuit: Circuit):
        """
        Estimate the cost of a circuit.

        Args:
            circuit(Circuit): Circuit to evaluate.

        Returns:
            float: Estimated cost of the circuit.
        """
        return -np.log(self.evaluate_fidelity(circuit))
