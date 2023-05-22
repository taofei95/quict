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

    def __init__(self, qubit_number, bench_func=None):
        self._qubit_number = qubit_number
        self._bench_func = bench_func

    # completely random circuits
    def _random_circuit(self):
        cir_random = Circuit(self._qubit_number)
        cir_random.random_append(self._qubit_number * 5)
        return cir_random

    # algorithm circuits
    def _alg_circuit(self):
        cir_alg = []
        alg_fields_list = ["adder", "clifford", "qft", "grover", "cnf", "maxcut", "qnn", "quantum_walk"]
        for field in alg_fields_list:
            cir = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=self._qubit_number)
            cir_alg.extend(cir)
        return cir_alg
    
    # random instruction set circuits
    def _inset_circuit(self):
        cir_inset = []
        random_fields_list = [
            "aspen-4", "ourense", "rochester", "sycamore", "tokyo", "ctrl_unitary", "diag",
            "single_bit", "ctrl_diag", "google", "ibmq", "ionq", "ustc", "nam", "origin"
            ]
        for field in random_fields_list:
            cir = CircuitLib().get_random_circuit(str(field), qubits_interval=self._qubit_number)
            cir_inset.extend(cir)
        return cir_inset

    # clifford circuits
    def _clifford_circuit(self):
        cir_cliff = Circuit(self._qubit_number)
        cir_cliff.random_append(self._qubit_number * 5, CLIFFORD_GATE_SET)
        return cir_cliff

    # circuits with different probabilities of cnot
    def _cx_circuit(self):
        cir_cx = []
        prob_list = [[1, 0], [0.8, 0.2], [0.6, 0.4], [0.4, 0.6], [0.2, 0.8]]
        for prob in prob_list:
            circuit = Circuit(self._qubit_number)
            circuit.random_append(self._qubit_number * 5, [GateType.cx, GateType.h], prob)
            cir_cx.append(circuit)
        return cir_cx

    # unitary matrix
    def _unitary_group(self):
        U = unitary_group.rvs(2 ** self._qubit_number)
        return U

    # general state vectors
    def _state_vector(self):
        state_vector = random_unit_vector(1 << self._qubit_number)
        return state_vector

    # error connection layout mapping circuit
    def _errorlayout_circuit(self):
        cir_map = []
        cir = Circuit(self._qubit_number)
        cir.random_append(self._qubit_number * 5)
        sub_layout = Layout(self._qubit_number)
        related_list = list(range(self._qubit_number))
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
        cir_map.append(cir1)
        return cir_map

    # different distance between cx gates
    

    def get_circuits(self):
        circuits_list = []
        if self._bench_func == "mapping":
            circuits_list.append(self._errorlayout_circuit())

        if self._bench_func == "optimizaiton":
            circuits_list.append(self._random_circuit())
            circuits_list.extend(self._alg_circuit())
            circuits_list.extend(self._inset_circuit())
            circuits_list.append(self._clifford_circuit())
            circuits_list.extend(self._cx_circuit())

            return circuits_list
        else:
            circuits_list.append(self._random_circuit())
            circuits_list.extend(self._alg_circuit())
            circuits_list.extend(self._inset_circuit())
            # circuits_list.append(self._unitary_group())
            # circuits_list.append(self._state_vector())

        return circuits_list

    def run(self, qcda_interface):
        circuits_list = self.get_circuits()
        circuits_bench = []
        for circuit in circuits_list:
            circuit_new = qcda_interface(circuit)
            circuits_bench.append(circuit_new)

        return circuits_bench

    def evaluate(self, circuits_list):
        bench_results = []
        for circuit in circuits_list:
            if self._bench_func == "mapping":
                bench_result = circuit.count_gate_by_gatetype(GateType.swap)
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
