import random

import pandas as pd
from QuICT.benchmark.benchcirdata import BenchCirData
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType
from QuICT.tools.circuit_library import CircuitLib


class QCDAbenchmark:
    """ A benchmarking framework for quantum circuits automated design."""

    def _get_alg_circuit(self, qubit_number, bench_func):
        cirs_group = []
        alg_fields_list = ["adder", "clifford", "qft", "grover", "cnf", "maxcut", "qnn", "quantum_walk", "vqe"]
        for field in alg_fields_list:
            cirs = CircuitLib().get_circuit("algorithm", str(field), qubits_interval=qubit_number)
            for cir in cirs:
                cir.gate_decomposition()
                cir.name = "+".join([bench_func, field, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def _get_machine_circuit(self, qubit_number, bench_func):
        cirs_group = []
        machine_fields_list = ["aspen-4", "ourense", "rochester", "sycamore", "tokyo"]
        for field in machine_fields_list:
            cirs = CircuitLib().get_circuit("machine", str(field), qubits_interval=qubit_number)
            for cir in cirs:
                cir.name = "+".join([bench_func, field, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def _random_prob_circuit(self, qubit_number, bench_func):
        cirs_group = []
        prob_list = [0.2, 0.4, 0.6, 0.8, 1]
        one_qubit = [GateType.h, GateType.rx, GateType.ry, GateType.rz, GateType.x, GateType.y, GateType.z, GateType.u3]
        two_qubits = [GateType.cx, GateType.cz, GateType.swap]

        for prob in prob_list:
            len_s, len_d = len(one_qubit), len(two_qubits)
            prob = [prob / len_s] * len_s + [(1 - prob) / len_d] * len_d
            cir = Circuit(qubit_number)
            cir.random_append(qubit_number * 10, typelist=one_qubit+two_qubits, probabilities=prob, seed=10)
            cir.name = "+".join([bench_func, "probrandom", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
            cirs_group.append(cir)
        return cirs_group

    def _special_layout_circuit(self, qubit_number, bench_func, layout, type):
        cirs_group = []
        layout_edges = []
        layout_edge = layout.edge_list
        for l in layout_edge:
            layout_edges.append([l.u, l.v])

        edges_all = list(range(qubit_number))
        edges_extra = []
        for i in edges_all:
            for j in edges_all:
                if i != j and [i, j] not in layout_edges:
                    edges_extra.append([i, j])

        if type == "extra_edges":
            cir = Circuit(qubit_number)
            gates = qubit_number * 10
            extra_edges = random.sample(edges_extra, int(random.uniform(0, len(edges_extra))))
            while cir.size() < gates:
                for edge in extra_edges:
                    CX | cir(edge)
                for _ in range(gates - len(extra_edges)):
                    edge = random.choice(layout_edges)
                    CX | cir(edge)
            cir.name = "+".join([bench_func, "prob", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
            cirs_group.append(cir)
        elif type == "optimal_break":
            cir = Circuit(qubit_number)
            gates = qubit_number * 10
            edges_error = random.sample(edges_extra, qubit_number)
            for n in edges_error:
                layout_edges.insert(int((len(layout_edges) / 2)), n)

            while cir.size() < gates:
                i = random.choice(layout_edges)
                CX | cir(i)
            cir.name = "+".join([bench_func, "nooptimal", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
            cirs_group.append(cir)
        elif type == "inverse_edges":
            cir = Circuit(qubit_number)
            gates = qubit_number * 10
            while cir.size() < gates:
                i = random.choice(edges_extra)
                CX | cir(i)
            cir.name = "+".join([bench_func, "inverse", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
            cirs_group.append(cir)

        return cirs_group

    def _clifford_pauli_circuit(self, qubit_number, bench_func):
        cirs_group = []
        for field in [CLIFFORD_GATE_SET, PAULI_GATE_SET]:
            cir = Circuit(qubit_number)
            cir.random_append(qubit_number * 5, field)
            cir.name = "+".join([bench_func, "special", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
            cirs_group.append(cir)
        return cirs_group

    def _template_circuit(self, qubit_number, bench_func):
        cirs_group = []
        cirs = CircuitLib().get_template_circuit(qubits_interval=qubit_number)
        for cir in cirs:
            sub_cir = cir.sub_circuit(qubit_limit=list(range(int(cir.width() / 2))))
            sub_cir | cir
            cir.name = "+".join([bench_func, "optimal", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
            cirs_group.append(cir)
        return cirs_group

    def _auto_compile_bench(self, qubit_number, bench_func):
        circuits_list = []
        # random probabilitity circuit
        circuits_list.extend(self._random_prob_circuit(qubit_number, bench_func))

        # algorithm circuit
        circuits_list.extend(self._get_alg_circuit(qubit_number, bench_func))

        # machine inset circuit
        circuits_list.extend(self._get_machine_circuit(qubit_number, bench_func))
        return circuits_list

    def _mapping_bench(self, qubit_number, layout, bench_func):
        circuits_list = []
        # algorithm circuit
        circuits_list.extend(self._get_alg_circuit(qubit_number, bench_func))

        # A probability distribution circuit that conforms to the topology
        circuits_list.extend(self._special_layout_circuit(qubit_number, bench_func, layout, "extra_edges"))

        # Approaching the known optimal mapping circuit
        circuits_list.extend(self._special_layout_circuit(qubit_number, bench_func, layout, "optimal_break"))

        # completely random circuits that conform to the inverse topology
        circuits_list.extend(self._special_layout_circuit(qubit_number, bench_func, layout, "inverse_edges"))
        return circuits_list

    def _optimization_bench(self, qubit_number, bench_func):
        circuits_list = []
        
        # algorithm circuit + instruction set circuit + circuits with different probabilities of cnot
        circuits_list.extend(self._auto_compile_bench(qubit_number, bench_func))

        # clifford / pauli instruction set circuit
        circuits_list.extend(self._clifford_pauli_circuit(qubit_number, bench_func))

        # Approaching the known optimal mapping circuit
        circuits_list.extend(self._template_circuit(qubit_number, bench_func))

        return circuits_list

    def _synthesis_bench(self, qubit_number, bench_func):
        # completely random circuits
        circuits_list = self._random_prob_circuit(qubit_number, bench_func)

        return circuits_list

    def get_circuits(
        self,
        VirtualQuantumMachine,
        bench_func:str = "auto"
        ):
        """Get the circuit to be benchmarked

        Args:
            VirtualQuantumMachine (VirtualQuantumMachine, optional): The information about the quantum machine.
            bench_func (str, optional): The type of qcdabenchmark. Defaults to "auto".

        Returns:
            (List[Circuit]): Return the list of output circuit order by output_type.
        """
        qubit_number = VirtualQuantumMachine.qubit_number
        layout = VirtualQuantumMachine.layout

        if bench_func == "auto":
            circuits_list = self._auto_compile_bench(qubit_number, bench_func)
        elif bench_func == "mapping":
            circuits_list = self._mapping_bench(qubit_number, layout, bench_func)
        elif bench_func == "optimization":
            circuits_list = self._optimization_bench(qubit_number, bench_func)
        elif bench_func == "synthesis":
            circuits_list = self._synthesis_bench(qubit_number, bench_func)
        return circuits_list

    def run(
        self,
        VirtualQuantumMachine,
        qcda_interface,
        bench_func:str = "auto"
    ):
        """Connect real-time benchmarking to the sub-physical machine to be measured.

        Args:
            VirtualQuantumMachine(VirtualQuantumMachine, optional): The information about the quantum machine.
            qcda_interface(optional): this is an interface that makes a series of optimizations to the original circuit 
                provided and returns the optimized circuit.
                input and output for example:

                def qcda_interface(circuit):
                    qcda_cir = optimizer(circuit)
                    return qcda_cir
            bench_func (str, optional): The type of qcdabenchmark. Defaults to "auto".
        Returns:
            Return the analysis of QCDAbenchmarking.
        """
        circuits_list = self.get_circuits(VirtualQuantumMachine, bench_func)
        ori_cir_bench, opt_cir_bench = [], []
        for circuit in circuits_list:
            circuit_new = qcda_interface(circuit)
            circuit_new.name = "+".join([
                circuit.name.split("+")[0],
                circuit.name.split("+")[1],
                f"w{circuit_new.width()}_s{circuit_new.size()}_d{circuit_new.depth()}"
            ])
            cir_opt = BenchCirData(circuit_new)
            opt_cir_bench.append(cir_opt)
            cir_ori = BenchCirData(circuit)
            ori_cir_bench.append(cir_ori)

        self.evaluate(bench_func, ori_cir_bench, opt_cir_bench)

    def evaluate(self, bench_func, ori_cir_bench, opt_cir_bench):
        self._benchmark_evaluate(bench_func, ori_cir_bench, opt_cir_bench)

    def _analysis_cir(self, ori_cir, opt_cir):
        result_dict = {
            "cir_type": ori_cir.field,
            "ori_size": ori_cir.size,
            "opt_size": opt_cir.size,
            "ori_depth": ori_cir.depth,
            "opt_depth": opt_cir.depth,
            "ori_one_qubit_gate": ori_cir.one_qubit_gate,
            "opt_one_qubit_gate": opt_cir.one_qubit_gate,
            "ori_two_qubit_gate": ori_cir.two_qubit_gate,
            "opt_two_qubit_gate": opt_cir.two_qubit_gate,
            "ori_swap_gate": ori_cir.swap_count_gate,
            "opt_swap_gate": opt_cir.swap_count_gate,
        }
        return result_dict

    def _benchmark_evaluate(self, bench_func, ori_cir_bench, opt_cir_bench):
        columns_list = [
            'cir_type', 'ori_size', 'opt_size', 'ori_depth', 'opt_depth', 'ori_one_qubit_gate', 'opt_one_qubit_gate', 
            'ori_two_qubit_gate', 'opt_two_qubit_gate', 'ori_swap_gate', 'opt_swap_gate'
        ]
        if bench_func == "optimization" or bench_func == "auto":
            results = columns_list[:5]
        elif bench_func == "mapping":
            results = columns_list[:5] + columns_list[-2:]
        elif bench_func == "synthesis":
            results = columns_list[:8]

        df = pd.DataFrame(columns=results)
        for i in range(len(ori_cir_bench)):
            result_list = []
            result = self._analysis_cir(ori_cir_bench[i], opt_cir_bench[i])
            columns_list = [i for i in result]
            for j in results:
                result_list.append(result[j])
            df.loc[i] = result_list
        print(df)





