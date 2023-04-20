import random
from abc import abstractmethod
import networkx as nx
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate, CompositeGate
from QuICT.core.utils import GateType


class Backend:
    def __init__(self,
                 n_qubit,
                 gate_set: list[GateType],
                 qubit_t1: list[float] = None,
                 qubit_t2: list[float] = None,
                 single_qubit_gate_fidelity: dict[int, float] = None,
                 two_qubit_gate_fidelity: dict[tuple, float] = None,
                 ):
        """
        Args:
            n_qubit(int): Number of qubits.
            gate_set(list[GateType]): List of gate types supported by the backend.
            qubit_t1(list[float]): List of T1 values for each qubit.
            qubit_t2(list[float]): List of T2 values for each qubit.
            single_qubit_gate_fidelity(dict[int, float]): Fidelity of single qubit gates.
            two_qubit_gate_fidelity(dict[tuple, float]): Fidelity of two qubit gates.
        """
        self.n_qubit = n_qubit
        self.gate_set = gate_set
        self.qubit_t1 = qubit_t1
        self.qubit_t2 = qubit_t2
        self.single_qubit_gate_fidelity = single_qubit_gate_fidelity
        self.two_qubit_gate_fidelity = two_qubit_gate_fidelity

    @abstractmethod
    def execute_circuit(self, circ: Circuit, n_shot: int, *args, **kwargs) -> dict[str, float]:
        """
        Execute a circuit on the backend.

        Args:
            circ(Circuit): Circuit to be executed.
            n_shot(int): Number of repeated executions.

        Returns:
            dict[str, float]: Output probability distribution
        """
        pass

    def _generate_random_circuit(self, n_qubit: int, n_gate: int):
        rand_circ = Circuit(n_qubit)
        rand_circ.random_append(n_gate, typelist=self.gate_set)

        circ = Circuit(n_qubit)
        two_qubit_gates = list(filter(
            lambda x: max(x[0], x[1]) < n_qubit,
            self.two_qubit_gate_fidelity.keys())
        )
        for g in rand_circ.gates:
            g: BasicGate
            if g.targets + g.controls == 1:
                g | circ(random.randint(0, n_qubit - 1))
            else:
                regs = list(random.choice(two_qubit_gates))
                g | circ(regs)
        return circ

    def _get_circuit_pst(self, circ: Circuit):
        new_circ = Circuit(circ.width())
        for g in circ.gates:
            new_circ.append(g)
        for g in reversed(circ.gates):
            g: BasicGate
            g.inverse() | new_circ(g.cargs + g.targs)

        prob_dist = self.execute_circuit(new_circ, 2000)
        if not prob_dist:
            return -1

        new_dist = {int(key, 2): val for key, val in prob_dist.items()}
        return new_dist[0] if 0 in new_dist else 0

    def _get_circost_tv(self, circ: Circuit):
        pass

    def generate_benchmark(self, n: int, max_qubit: int, max_gate: int):
        """
        Generate benchmark random circuits.

        Args:
            n(int): Number of benchmark circuits.
            max_qubit(int): Maximum number of qubits.
            max_gate(int): Maximum number of gates.

        Returns:
            list[(Circuit, float)]: List of benchmark circuits and their corresponding probability of success.
        """
        bmks = []
        for _ in range(n):
            n_qubit = np.random.randint(2, max_qubit + 1)
            n_gate = np.random.randint(1, max_gate + 1)
            circ = self._generate_random_circuit(n_qubit, n_gate)
            pst = self._get_circuit_pst(circ)
            # print(f'bmk {_}: n_qubit={n_qubit}, n_gate={n_gate}, pst={pst}')
            if pst >= 0:
                bmks.append((circ, pst))
        return bmks

    def estimated_cost(self, circ):
        """
        Estimate the cost of a circuit.

        Args:
            circ(Circuit): Circuit to be estimated.

        Returns:
            float: Estimated cost.
        """

        cost = 0
        for g in circ.gates:
            gate_f = 1
            if g.controls + g.targets == 1:
                if self.single_qubit_gate_fidelity:
                    gate_f = self.single_qubit_gate_fidelity[g.targ]
            else:
                key = tuple(g.cargs + g.targs)
                if key in self.two_qubit_gate_fidelity:
                    gate_f = self.two_qubit_gate_fidelity[key]

            avg_time = 0
            for q in g.cargs + g.targs:
                avg_time += self.qubit_t1[q] + self.qubit_t2[q]
            avg_time /= (g.controls + g.targets)

            # print(f'gate_f={-np.log(gate_f)}, avg_time={avg_time}')

            # g_cost = (-np.log(gate_f) + 1) / avg_time
            # g_cost = (-np.log(gate_f) * 100 + 1) / avg_time

            g_cost = (-np.log(gate_f) + 1) / avg_time
            # print(g_cost)
            cost += g_cost
        return cost
