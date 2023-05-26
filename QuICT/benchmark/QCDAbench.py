import os
import random
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout
from scipy.stats import unitary_group
from QuICT.qcda.mapping.mcts.mcts_mapping import MCTSMapping
from QuICT.tools.circuit_library.circuitlib import CircuitLib
from unit_test.qcda.synthesis.quantum_state_preparation.quantum_state_preparation_unit_test import random_unit_vector
from prettytable import PrettyTable


class QCDAbench:

    def __init__(self, bench_func: str = "mapping"):
        self._bench_func = bench_func

    def __mapping_bench(self, qubit_number, layout_edges):
        cirs_group = []

        # algorithm circuit
        alg_fields_list = ["adder", "clifford", "qft", "grover", "cnf", "maxcut", "qnn", "quantum_walk"]
        for field in alg_fields_list:
            cir = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=qubit_number)
            cirs_group.extend(cir)

        # A probability distribution circuit that conforms to the topology
        edges_list = layout_edges
        edges_all = list(range(qubit_number))
        edges_new = []
        for i in edges_all:
            for j in edges_all:
                if i != j and [i, j] not in edges_list:
                    edges_new.append([i, j])

        cir = Circuit(qubit_number)
        gates = qubit_number * 10
        prob = [0.2, 0.4, 0.6, 0.8]
        for p in prob:
            while cir.size() < qubit_number * 10:
                size = int(p * gates)
                while cir.size() < size:
                    extra_number = size - cir.size()
                    if extra_number >= len(edges_list):
                        for edge in edges_list:
                            CX | cir(edge)
                    else:
                        for i in range(extra_number):
                            CX | cir(edges_list[i])
                i = random.choice(edges_new)
                CX | cir(i)
        cirs_group.append(cir)

        # Approaching the known optimal mapping circuit
        # 


        # completely random circuits that conform to the inverse topology
        edges_list = layout_edges
        edges_all = list(range(qubit_number))
        edges_new = []

        for i in edges_all:
            for j in edges_all:
                if i != j and [i, j] not in edges_list:
                    edges_new.append([i, j])

        cir = Circuit(qubit_number)
        gates = qubit_number * 10
        while cir.size() < gates:
            i = random.choice(edges_new)
            CX | cir(i)
        cirs_group.append(cir)

        # error connection layout mapping circuit
        cir = Circuit(qubit_number)
        cir.random_append(qubit_number * 5)
        sub_layout = Layout(qubit_number)
        related_list = list(range(qubit_number))
        n = 0
        edges_list = []
        while n < len(related_list) - 1:
            edges = (related_list[n], related_list[n + 1])
            edges_list.append(edges)
            n += 1
            related_list.remove(random.choice(related_list))
        for u, v in edges_list:
            sub_layout.add_edge(u, v)

        mcts = MCTSMapping(sub_layout)
        cir1 = mcts.execute(cir)
        cirs_group.append(cir1)

    def __optimization_bench(self, qubit_number):
        cirs_group = []

        # algorithm circuit
        alg_fields_list = ["adder", "clifford", "qft", "grover", "cnf", "maxcut", "qnn", "quantum_walk"]
        for field in alg_fields_list:
            cir = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=qubit_number)
            cirs_group.extend(cir)

        # instruction set circuit
        random_fields_list = [
            "aspen-4", "ourense", "rochester", "sycamore", "tokyo", "ctrl_unitary", "diag",
            "single_bit", "ctrl_diag", "google", "ibmq", "ionq", "ustc", "nam", "origin"
            ]
        for field in random_fields_list:
            cir = CircuitLib().get_random_circuit(str(field), qubits_interval=qubit_number)
            cirs_group.extend(cir)

        # clifford / pauil instruction set circuit
        for field in [CLIFFORD_GATE_SET, PAULI_GATE_SET]:
            cir = Circuit(qubit_number)
            cir.random_append(qubit_number * 5, CLIFFORD_GATE_SET)
            cirs_group.extend(cir)

        # circuits with different probabilities of cnot
        prob_list = [[1, 0], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]]
        for prob in prob_list:
            cir = Circuit(qubit_number)
            cir.random_append(qubit_number * 10, [GateType.cx, GateType.h], prob)
            cirs_group.append(cir)

        return cirs_group

    def __synthesis_bench(self, qubit_number):
        cirs_group = []

        # completely random circuits
        cir = Circuit(qubit_number)
        for prob in [5, 10, 15, 20]:
            cir.random_append(qubit_number * prob)
            cirs_group.append(cir)

        # different matrix of unitary
        U = unitary_group.rvs(2 ** qubit_number)

        # general and sparse quantum states
        state_vector = random_unit_vector(1 << qubit_number)

        return cirs_group


    def get_circuits(
        self,
        VirtualQuantumMachine,
        ):
        """Get the circuit to be benchmarked

        Args:
            VirtualQuantumMachine(VirtualQuantumMachine, optional): The information about the quantum machine.

        Returns:
            (List[Circuit]): Return the list of output circuit order by output_type.
        """
        qubit_number = VirtualQuantumMachine.qubit_number
        layout = VirtualQuantumMachine.layout
        layout_edges = layout.edge_list

        if self._bench_func == "mapping":
            circuits_list = self.__mapping_bench(qubit_number, layout_edges)

        elif self._bench_func == "optimizaiton":
            circuits_list = self.__optimization_bench(qubit_number)

        else:
            circuits_list = self.__synthesis_bench(qubit_number)

        return circuits_list

    def run(self, VirtualQuantumMachine, qcda_interface):
        """Connect real-time benchmarking to the sub-physical machine to be measured.

        Args:
            VirtualQuantumMachine(VirtualQuantumMachine, optional): The information about the quantum machine.
            qcda_interface(optional): this is an interface that makes a series of optimizations to the original circuit 
                provided and returns the optimized circuit.
                input and output for example:

                def qcda_interface(circuit):
                    optimizer(circuit)
                    return qcda_cir

        Returns:
            Return the analysis of benchmarking.
        """
        circuits_list = self.get_circuits(VirtualQuantumMachine)
        circuits_bench = []
        for circuit in circuits_list:
            circuit_new = qcda_interface(circuit)
            circuits_bench.append(circuit_new)
        self.evaluate(circuits_bench)

    def evaluate(self, circuits_list):
        bench_results = []
        for circuit in circuits_list:
            if self._bench_func == "mapping":
                bench_result = [
                    circuit.size(),
                    circuit.depth(),
                    circuit.count_gate_by_gatetype(GateType.swap)
                ]
                bench_results.append(bench_result)
                table = PrettyTable(['size', 'swap gates'])
                for i in range(len(bench_results)):
                    table.add_row(bench_results[i])
                return table
            else:
                bench_result = [
                    circuit.width(),
                    circuit.size(),
                    circuit.depth(),
                    circuit.count_1qubit_gate(),
                    circuit.count_2qubit_gate()
                ]
                bench_results.append(bench_result)

                table = PrettyTable(['width', 'size', 'depth', '1qubit gate', '2qubit gate'])
                for i in range(len(bench_results)):
                    table.add_row(bench_results[i])
                return table
