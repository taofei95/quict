import os
import random
import re
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType
from QuICT.tools.circuit_library import CircuitLib
from scipy.stats import unitary_group
from unit_test.qcda.synthesis.quantum_state_preparation.quantum_state_preparation_unit_test import random_unit_vector
from prettytable import PrettyTable


class QCDAbench:

    def __init__(self, bench_func: str = "mapping"):
        self._bench_func = bench_func

    def _mapping_bench(self, qubit_number, layout_edges):
        cirs_group = []

        # algorithm circuit
        alg_fields_list = ["adder", "clifford", "qft", "grover", "cnf", "maxcut", "qnn", "quantum_walk", "vqe"]
        for field in alg_fields_list:
            cir = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=[qubit_number])
            for c in cir:
                c.gate_decomposition()
                c.name = self._bench_func + "+" + field
                cirs_group.append(c)

        # A probability distribution circuit that conforms to the topology
        edges_list = layout_edges
        edges_all = list(range(qubit_number))
        edges_extra = []
        for i in edges_all:
            for j in edges_all:
                if i != j and [i, j] not in edges_list:
                    edges_extra.append([i, j])

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
                i = random.choice(edges_extra)
                CX | cir(i)
            cir.name = self._bench_func + "+" + "prob"
            cirs_group.append(cir)

        # Approaching the known optimal mapping circuit
        edges_list = layout_edges
        edges_all = list(range(qubit_number))
        edges_extra = []

        for i in edges_all:
            for j in edges_all:
                if i != j and [i, j] not in edges_list:
                    edges_extra.append([i, j])

        cir = Circuit(qubit_number)
        gates = qubit_number * 10
        edges_error = random.sample(edges_extra, qubit_number)
        for n in edges_error:
            edges_list.insert(int((len(edges_list) / 2)), n)

        while cir.size() < gates:
            extra_number = gates - cir.size()
            if extra_number >= len(edges_list):
                for edge in edges_list:
                    CX | cir(edge)
            else:
                for i in range(extra_number):
                    CX | cir(edges_list[i])
        cir.name = self._bench_func + "+" + "optimal"
        cirs_group.append(cir)

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
        cir.name = self._bench_func + "+" + "inverse"
        cirs_group.append(cir)

        return cirs_group

    def _optimization_bench(self, qubit_number):
        cirs_group = []

        # algorithm circuit
        alg_fields_list = ["adder", "clifford", "qft", "grover", "cnf", "maxcut", "qnn", "quantum_walk"]
        for field in alg_fields_list:
            cir = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=[qubit_number])
            for c in cir:
                c.name = self._bench_func + "+" + field
            cirs_group.extend(cir)

        # instruction set circuit
        random_fields_list = [
            "aspen-4", "ourense", "rochester", "sycamore", "tokyo", "ctrl_unitary", "diag",
            "single_bit", "ctrl_diag", "google", "ibmq", "ionq", "ustc", "nam", "origin"
            ]
        for field in random_fields_list:
            cir = CircuitLib().get_random_circuit(str(field), qubits_interval=[qubit_number])
            for c in cir:
                c.name = self._bench_func + "+" + "inset"
            cirs_group.extend(cir)

        # clifford / pauil instruction set circuit
        for field in [CLIFFORD_GATE_SET, PAULI_GATE_SET]:
            cir = Circuit(qubit_number)
            cir.random_append(qubit_number * 5, field)
            cir.name = self._bench_func + "+" + "special"
            cirs_group.append(cir)

        # circuits with different probabilities of cnot
        prob_list = [[1, 0], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]]
        for prob in prob_list:
            cir = Circuit(qubit_number)
            cir.random_append(qubit_number * 10, [GateType.cx, GateType.h], prob)
            cir.name = self._bench_func + "+" + "prob"
            cirs_group.append(cir)

        # Approaching the known optimal mapping circuit
        cir = CircuitLib().get_template_circuit(qubits_interval=qubit_number)
        for c in cir:
            sub_cir = c.sub_circuit(qubit_limit=list(range(int(c.width() / 2))))
            for cgate in sub_cir.to_compositegate():
                cgate.inverse() | c
            c.name = self._bench_func + "+" + "optimal"
            cirs_group.append(c)
        
        return cirs_group

    def _synthesis_bench(self, qubit_number):
        # completely random circuits
        random_cirs = []
        for prob in [5, 10, 15, 20]:
            cir = Circuit(qubit_number)
            cir.random_append(qubit_number * prob)
            cir.name = self._bench_func + "+" + "random"
            random_cirs.append(cir)

        # different matrix of unitary
        q_nums = random.sample(list(range(qubit_number))[1:], 4)
        for q in q_nums:
            U = unitary_group.rvs(2 ** q)
            random_cirs.append(U)

            # general and sparse quantum states
            state_vector = random_unit_vector(1 << q)
            random_cirs.append(state_vector)

        return random_cirs

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
        layout_edge = layout.edge_list
        layout_edges = []
        for l in layout_edge:
            layout_edges.append([l.u, l.v])

        if self._bench_func == "mapping":
            circuits_list = self._mapping_bench(qubit_number, layout_edges)

        elif self._bench_func == "optimization":
            circuits_list = self._optimization_bench(qubit_number)
        else:
            circuits_list = self._synthesis_bench(qubit_number)

        return circuits_list

    def run(self, VirtualQuantumMachine, qcda_interface):
        """Connect real-time benchmarking to the sub-physical machine to be measured.

        Args:
            VirtualQuantumMachine(VirtualQuantumMachine, optional): The information about the quantum machine.
            qcda_interface(optional): this is an interface that makes a series of optimizations to the original circuit 
                provided and returns the optimized circuit.
                input and output for example:

                def qcda_interface(circuit):
                    qcda_cir = optimizer(circuit)
                    return qcda_cir

        Returns:
            Return the analysis of QCDAbenchmarking.
        """
        circuits_list = self.get_circuits(VirtualQuantumMachine)
        circuits_bench = []
        if self._bench_func != "synthesis":
            for circuit in circuits_list:
                circuit_new = qcda_interface(circuit)
                circuits_bench.append(circuit_new)
        else:
            circuits_bench = qcda_interface(circuits_list)

        self.evaluate(circuits_list, circuits_bench)

    def evaluate(self, circuits_list, circuits_bench):
        bench_results = []
        if self._bench_func == "optimization":
            for i in range(len(circuits_list)):
                bench_result = [
                    circuits_list[i].name,
                    circuits_list[i].width(),
                    circuits_bench[i].width(),
                    circuits_list[i].size(),
                    circuits_bench[i].size(),
                    circuits_list[i].depth(),
                    circuits_bench[i].depth(),
                    circuits_list[i].count_1qubit_gate(),
                    circuits_bench[i].count_1qubit_gate(),
                    circuits_list[i].count_2qubit_gate(),
                    circuits_bench[i].count_2qubit_gate()
                ]
                bench_results.append(bench_result)
            table = PrettyTable([
                'name', 'width', 'opt width', 'size', 'opt size', 'depth', 'opt depth', 
                '1qubit gate', 'opt 1qubit gate', '2qubit gate', 'opt 2qubit gate'
                ])
        elif self._bench_func == "mapping":
            for i in range(len(circuits_list)):
                bench_result = [
                    circuits_list[i].name,
                    circuits_list[i].width(),
                    circuits_bench[i].width(),
                    circuits_list[i].size(),
                    circuits_bench[i].size(),
                    circuits_list[i].depth(),
                    circuits_bench[i].depth(),
                    circuits_list[i].count_gate_by_gatetype(GateType.swap),
                    circuits_bench[i].count_gate_by_gatetype(GateType.swap)
                ]
                bench_results.append(bench_result)
            table = PrettyTable([
                'name', 'width', 'opt width', 'size', 'opt size', 'depth', 'opt depth', 'swap gate', 'opt swap gate'
                ])
        elif self._bench_func == "synthesis":
            for i in range(4):
                bench_result = [
                    circuits_list[i].name,
                    circuits_list[i].width(),
                    circuits_bench[i].width(),
                    circuits_list[i].size(),
                    circuits_bench[i].size(),
                    circuits_list[i].depth(),
                    circuits_bench[i].depth()
                ]
                bench_results.append(bench_result)
            table = PrettyTable([
                'name', 'width', 'opt width', 'size', 'opt size', 'depth', 'opt depth'
                ])

            for i in [4, 5, 6, 7]:
                origin_matrix = circuits_list[i]
                decom_matrix = circuits_bench[i].matrix
                print(f'origin_matrix:', origin_matrix)
                print(f'decom_matrix:', decom_matrix)
            for j in [8, 9, 10, 11]:
                origin_sv = circuits_list[j]
                amp_result = circuits_bench[j]
                print(f'origin_amplitude:', origin_sv)
                print(f'qsp_amplitude:', amp_result)

        for i in range(len(bench_results)):
            table.add_row(bench_results[i])
        print(table)
