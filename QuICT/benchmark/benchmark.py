import os
import random
import re
from typing import List
import pandas as pd
import prettytable as pt
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from QuICT.benchmark.benchlib import BenchLib
from QuICT.core.gate.gate import *
from QuICT.core.virtual_machine.virtual_machine import VirtualQuantumMachine

from QuICT.qcda.qcda import QCDA
from QuICT.core import Circuit
from QuICT.tools.circuit_library.circuitlib import CircuitLib
from QuICT.tools.circuit_library.get_benchmark_circuit import BenchmarkCircuitBuilder


class QuantumMachinebenchmark:
    """ The QuICT Benchmarking. """
    __alg_fields_list = ["qft", "adder", "cnf", "qnn", "quantum_walk"]

    def __init__(
        self,
        output_path: str = "./benchmark",
        output_file_type: str = "txt"
    ):
        """
        Initial benchmark for QuICT

        Args:
            output_path (str, optional): The path of the Analysis of the results.
            show_type (str, optional): Analysis of the Graph exists by default,
                and other analysis selects "txt" or "excel".
        """
        self._output_path = os.path.abspath(output_path)
        self._output_file_type = output_file_type

    def _get_random_circuit(self, level: int, q_number: int, Ins_set, layout=None):
        cir_list = []
        if level == 1:
            gate_prob, pro_s = range(2, 6), 0.9
        elif level == 2:
            gate_prob, pro_s = range(6, 10), 0.8
        elif level == 3:
            gate_prob, pro_s = range(10, 14), 0.7

        for gates in gate_prob:
            cir = Circuit(q_number)
            # Single-qubit gates
            size_s = int(q_number * gates * pro_s)
            cir.random_append(size_s, Ins_set.one_qubit_gates)

            # Double-qubits gates
            size_d = q_number * gates - size_s
            layout_list = layout.edge_list
            for _ in range(size_d):
                biq_gate = gate_builder(Ins_set.two_qubit_gate, random_params=True)
                bgate_layout = np.random.choice(layout_list)
                biq_gate | cir([bgate_layout.u, bgate_layout.v])

            # Build mirror circuit
            inverse_gate = cir.to_compositegate().inverse()
            inverse_gate | cir
            Measure | cir
            cir.name = "+".join(["random", "random", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}", f"level{level}"])
            cir_list.append(cir)
        return cir_list

    def _get_algorithm_circuit(self, vqm, level: int, enable_qcda_for_alg_cir):
        cir_list = []
        if level == 1:
            return []
        if level == 2:
            field = self.__alg_fields_list[:3]
        elif level == 3:
            field = self.__alg_fields_list

        if enable_qcda_for_alg_cir:
            qcda = QCDA()

        for i in range(len(field)):
            cirs = CircuitLib().get_circuit("algorithm", field[i], qubits_interval=vqm.qubit_number)
            for cir in cirs:
                if enable_qcda_for_alg_cir:
                    cir = qcda.auto_compile(cir, vqm)
                    Measure | cir
                else:
                    Measure | cir
                cir.name = "+".join([
                    "algorithm", field[i], f"w{cir.width()}_s{cir.size()}_d{cir.depth()}", f"level{level}"
                ])
                cir_list.append(cir)

        return cir_list

    def _get_benchmark_circuit(self, level: int, q_number: int, Ins_set):
        cir_list = []
        cirs = BenchmarkCircuitBuilder().get_benchmark_circuit(q_number, level, Ins_set)
        for cir in cirs:
            split = cir.name.split("+")
            attribute = re.findall(r'\d+(?:\.\d+)?', split[2])
            field, void_gates = split[1], attribute[3]
            inverse_cgate = cir.to_compositegate().inverse()
            inverse_cgate | cir
            if field != "mediate_measure":
                Measure | cir
            cir.name = "+".join([
                "benchmark", field, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}_v{void_gates}", f"level{level}"
            ])
            cir_list.append(cir)

        return cir_list

    def get_circuits(
        self,
        quantum_machine_info: VirtualQuantumMachine,
        level: int = 1,
        enable_qcda_for_alg_cir: bool = False
    ):
        """
        Get circuit from CircuitLib and Get the algorithm circuit after qcda.

        Args:
            quantum_machine_info(VirtualQuantumMachine, optional): The information about the quantum machine.
            level (int): Get the type of benchmark circuit group, include different circuits, one of [1, 2, 3],
                default 1.
            enable_qcda_for_alg_cir(bool): Auto-Compile the circuit with the given quantum machine info, just for
                algorithm circuit, default False.

        Returns:
            (List[Circuit]): Return the list of output circuit order by output_type.
        """

        # obey instruction set and layout in vqm to build random circuit
        q_number = quantum_machine_info.qubit_number
        Ins_set = quantum_machine_info.instruction_set
        layout = quantum_machine_info.layout

        # get circuits from circuitlib
        circuit_list = []

        # get random circuits
        circuit_list.extend(self._get_random_circuit(level, q_number, Ins_set, layout))

        # get benchmark circuits
        circuit_list.extend(self._get_benchmark_circuit(level, q_number, Ins_set))

        # get algorithm circuit
        circuit_list.extend(self._get_algorithm_circuit(quantum_machine_info, level, enable_qcda_for_alg_cir))

        # list of Special defined Circuit-benchmark data structure
        bench_cir = []
        for i in range(len(circuit_list)):
            benchlib = BenchLib(circuit_list[i])
            bench_cir.append(benchlib)

        return bench_cir

    def run(
        self,
        simulator_interface,
        quantum_machine_info: VirtualQuantumMachine,
        level: int = 1,
        enable_qcda_for_alg_cir: bool = False
    ):
        """
        Connect real-time benchmarking to the sub-physical machine to be measured.

        Args:
            simulator_interface(optional): Interface for the sub-physical machine to be measured, that is a function for
                realize the output quantum physics machine amplitude of the input circuit, saving circuit and amplitude
                input and output for example:

                def sim_interface(circuit):
                    simulation(circuit)
                    return amplitude

            quantum_machine_info(VirtualQuantumMachine, optional): (VirtualQuantumMachine, optional):
                The information about the quantum machine.
            level (int): Get the type of benchmark circuit group, include different circuits, one of [1, 2, 3],
                default 1.
            enable_qcda_for_alg_cir(bool): Auto-Compile the circuit with the given quantum machine info, just for
                algorithm circuit, default False.

        Returns:
            Return the analysis of benchmarking.
        """
        # Step1 : get circuits from circuitlib
        bench_cir = self.get_circuits(quantum_machine_info, level, enable_qcda_for_alg_cir)

        # Step 2: physical machine simulation
        for i in range(len(bench_cir)):
            sim_result = simulator_interface(bench_cir[i].circuit)
            bench_cir[i].machine_amp = sim_result

        # Step 3: evaluate each circuit
        self.evaluate(bench_cir)

        # Step 4: show result
        self.show_result(bench_cir)

    def evaluate(self, circuits: Union[List[BenchLib], BenchLib]):
        """ Evaluate all circuits in circuit list group by fields. """
        for bench_cir in circuits:
            if bench_cir.type == "random":
                self._evaluate_random_circuits(bench_cir)
            elif bench_cir.type == "benchmark":
                self._evaluate_benchmark_circuit(bench_cir)
            elif bench_cir.type == "algorithm":
                self._evaluate_algorithm_circuits(bench_cir)

    def _evaluate_random_circuits(self, bench_cir):
        cir_qv = bench_cir.qv
        cir_fidelity = bench_cir.fidelity
        cir_score = round(cir_qv * cir_fidelity, 4)
        bench_cir.benchmark_score = cir_score

    def _evaluate_benchmark_circuit(self, bench_cir):
        cir_qv = bench_cir.qv
        cir_fidelity = bench_cir.fidelity
        cir_value = bench_cir.value
        cir_score = round(cir_qv * cir_fidelity * cir_value, 4)
        bench_cir.benchmark_score = cir_score

    def _evaluate_algorithm_circuits(self, bench_cir):
        cir_qv = bench_cir.qv
        cir_fidelity = bench_cir.fidelity
        cir_score = round(cir_qv * cir_fidelity, 4)
        bench_cir.benchmark_score = cir_score

    def show_result(self, bench_cir):
        """ show benchmark result. """
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)

        if len(bench_cir) > 0:
            self._graph_show(bench_cir)

        if self._output_file_type == "txt":
            self._txt_show(bench_cir)

        else:
            self._excel_show(bench_cir)

    def _graph_show(self, bench_cir):
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('ggplot')
        ################################ random circuits benchmark #####################################
        # Construct the data
        feature = ['random1', 'random2', 'random3', 'random3']

        values = []
        for i in range(len(bench_cir)):
            field = bench_cir[i].field
            if field == "random":
                values.append(bench_cir[i].benchmark_score)

        N = len(values)
        random_data = values

        # Sets the angle of the radar chart to bisect a plane
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        feature = np.concatenate((feature, [feature[0]]))
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        # Draw the first diagram
        plt.figure(figsize=(12, 10), dpi=100)
        plt.figure(1.8)
        ax = plt.subplot(221, polar=True)
        ax.plot(angles, values, 'lightgreen', linewidth=2)
        ax.fill(angles, values, 'c', alpha=0.5)

        ax.set_thetagrids(angles * 180 / np.pi, feature)
        ax.set_ylim(0, np.floor(values.max()) + 2)

        plt.tick_params(labelsize=12)
        plt.title('Quantum machine inset circuits radar chart show')
        plt.legend(["score"])
        ax.grid(True)

        ################################ special circuits benchmark #####################################
        # Construct the data
        feature_name = ['parallelized', 'entangled', 'serialized', 'measure']

        P, E, S, M, values_1, feature_1 = [], [], [], [], [], []
        for i in range(len(bench_cir)):
            field = bench_cir[i].field
            if field == "highly_parallelized":
                P.append(bench_cir[i].benchmark_score)
            elif field == "highly_serialized":
                S.append(bench_cir[i].benchmark_score)
            elif field == "highly_entangled":
                E.append(bench_cir[i].benchmark_score)
            elif field == "mediate_measure":
                M.append(bench_cir[i].benchmark_score)

        for x in [P, E, S, M]:
            if len(x) > 0:
                values_1.append(max(x))
                feature_1.append(feature_name[[P, E, S, M].index(x)])
        N = len(values_1)
        special_data = values_1
        # Sets the angle of the radar chart to bisect a plane
        angles_1 = np.linspace(0, 2 * np.pi, N, endpoint=False)
        feature_1 = np.concatenate((feature_1, [feature_1[0]]))
        values_1 = np.concatenate((values_1, [values_1[0]]))
        angles_1 = np.concatenate((angles_1, [angles_1[0]]))

        # Draw the first diagram
        ax1 = plt.subplot(222, polar=True)
        ax1.plot(angles_1, values_1, 'y-', linewidth=2)
        ax1.fill(angles_1, values_1, 'r', alpha=0.5)

        ax1.set_thetagrids(angles_1 * 180 / np.pi, feature_1)
        ax1.set_ylim(0, np.floor(values_1.max()) + 0.5)

        plt.tick_params(labelsize=12)
        plt.title('Special benchmark circuits radar chart show')
        plt.legend(["score"])

        ax1.grid(True)

        ################################### algorithmic circuits benchmark ##############################
        # Construct the data
        value_list, feature, values_2 = [], [], []
        field_list = ["qft", "adder", "cnf", "vqe", "qnn", "quantum_walk"]
        for i in range(len(bench_cir)):
            field = bench_cir[i].field
            if field in field_list:
                QV = bench_cir[i].qv
                value_list.append([field, QV])
        if len(value_list) > 0:
            field_QV_map = defaultdict(list)
            for field, QV in value_list:
                field_QV_map[field].append(QV)
                feature.append(field)
            feature_2 = list(set(feature))
            feature_2.sort(key=feature.index)
            for value in feature_2:
                values_2.append(max(field_QV_map[value]))
            # Sets the angle of the radar chart to bisect a plane
            N = len(values_2)
            if N > 4:
                alg_data = random.sample(list(values_2), 4)
            elif N == 4:
                alg_data = values_2
            elif N < 4:
                alg_data = values_2 + [0] * (4 - len(values_2))
            angles_2 = np.linspace(0, 2 * np.pi, N, endpoint=False)
            feature_2 = np.concatenate((feature_2, [feature_2[0]]))
            values_2 = np.concatenate((values_2, [values_2[0]]))
            angles_2 = np.concatenate((angles_2, [angles_2[0]]))
            # Draw the second diagram
            ax2 = plt.subplot(223, polar=True)
            ax2.plot(angles_2, values_2, 'c-', linewidth=2)
            ax2.fill(angles_2, values_2, 'b', alpha=0.5)
            ax2.set_thetagrids(angles_2 * 180 / np.pi, feature_2)
            ax2.set_ylim(0, np.floor(values_2.max()) + 0.5)

            plt.tick_params(labelsize=12)
            plt.title('Algorithmic circuits benchmark radar chart show')
            plt.legend(["score"])

            ax2.grid(True)
        else:
            alg_data = values_2 + [0] * (4 - len(values_2))

        ################################ the overall benchmark score #####################################
        radar_labels = np.array(['random', 'special', 'algorithm'])
        nAttr = 3
        data = np.array([
            list(random_data),
            list(special_data),
            alg_data
        ])
        angles_3 = np.linspace(0, 2 * np.pi, nAttr, endpoint=False)
        data = np.concatenate((data, [data[0]]))
        angles_3 = np.concatenate((angles_3, [angles_3[0]]))
        ax3 = plt.subplot(224, polar=True)
        ax3.plot(angles_3, data, 'bo-', color='gray', linewidth=1, alpha=0.2)
        ax3.plot(angles_3, data, 'o-', linewidth=1.5, alpha=0.2)
        ax3.fill(angles_3, data, alpha=0.25)
        plt.thetagrids((angles_3 * 180 / np.pi)[:-1], radar_labels)
        plt.title('The overall benchmark score radar chart show')
        plt.legend(["number type score"])
        ax3.set_ylim(0, np.floor(data.max()) + 4)
        plt.grid(True)

        plt.savefig(self._output_path + "/benchmark_radar_chart_show.jpg")
        plt.show()

    def _txt_show(self, bench_cir):
        result_file = open(self._output_path + '/benchmark_txt_show.txt', mode='w+', encoding='utf-8')
        tb = pt.PrettyTable()
        tb.field_names = [
            'field', 'circuit width', 'circuit size', 'circuit depth', 'fidelity', 'quantum volume', 'benchmark score'
        ]
        for i in range(len(bench_cir)):
            cir = bench_cir[i]
            tb.add_row([cir.field, cir.width, cir.size, cir.depth, cir.fidelity, cir.qv, cir.benchmark_score])
        result_file.write(str(tb))
        result_file.close()

    def _excel_show(self, bench_cir):
        dfData_list = []
        for i in range(len(bench_cir)):
            cir = bench_cir[i]
            dfData = {
                'field': cir.field,
                'circuit width': cir.width,
                'circuit size': cir.size,
                'circuit depth': cir.depth,
                'fidelity': cir.fidelity,
                'quantum volume': cir.qv,
                'benchmark score': cir.benchmark_score
            }
            dfData_list.append(dfData)

        df = pd.DataFrame(dfData_list)
        df.to_excel(self._output_path + "/benchmark_excel_show.xlsx")
