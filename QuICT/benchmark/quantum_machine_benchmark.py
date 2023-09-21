import os
import random
import re
import pandas as pd
import prettytable as pt
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from QuICT.core import Circuit
from QuICT.core.gate import Measure
from QuICT.core.virtual_machine import VirtualQuantumMachine
from QuICT.qcda import QCDA
from QuICT.tools.circuit_library.circuitlib import CircuitLib

from .benchcirdata import BenchCirData
from .get_benchmark_circuit import BenchmarkCircuitBuilder


class QuantumMachinebenchmark:
    """ The quantum machine Benchmark. """
    __alg_fields_list = ["adder", "qft", "cnf", "qnn", "vqe", "quantum_walk"]

    def __init__(
        self,
        output_path: str = "./benchmark",
        output_file_type: str = "txt"
    ):
        """
        Initial benchmark for QuICT

        Args:
            output_path (str, optional): The path of the Analysis of the results.
            output_file_type (str, optional): Analysis of the Graph exists by default,
                and other analysis selects "txt" or "excel".
        """
        self._output_path = os.path.abspath(output_path)
        self._output_file_type = output_file_type

    def _get_random_circuit(self, level: int, q_number: int, ins_set, layout, is_measure):
        cir_list = []
        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4)
        pro_s = level / 10

        len_s, len_d = len(ins_set.one_qubit_gates), len([ins_set.two_qubit_gate])
        prob = [(1 - pro_s) / len_s] * len_s + [pro_s / len_d] * len_d

        gate_type = ins_set.one_qubit_gates + [ins_set.two_qubit_gate]

        for gates in gate_prob:
            cir = Circuit(wires=q_number, topology=layout)
            cir_size = q_number * gates
            cir.random_append(rand_size=cir_size, typelist=gate_type, random_params=True, probabilities=prob)
            # Build mirror circuit
            cir.inverse() | cir
            if is_measure:
                Measure | cir
            cir.name = "+".join(["random", "random", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}", f"level{level}"])
            cir = BenchCirData(cir)
            cir_list.append(cir)
        return cir_list

    def _get_algorithm_circuit(self, vqm, level: int, enable_qcda_for_alg_cir, is_measure):
        cir_list, alg_cirs = [], []
        if level == 1:
            return []
        if level == 2:
            field = self.__alg_fields_list[:3]
        else:
            field = self.__alg_fields_list[:]

        if enable_qcda_for_alg_cir:
            qcda = QCDA()

        for i in range(len(field)):
            cirs = CircuitLib().get_circuit("algorithm", field[i], qubits_interval=[vqm.qubit_number, vqm.qubit_number])
            alg_cirs.extend(cirs)
        for cir in alg_cirs:
            field = cir.name.split("+")[0]
            if enable_qcda_for_alg_cir is True:
                if field != "vqe":
                    cir = qcda.auto_compile(cir, vqm)
            if is_measure:
                Measure | cir
            cir.name = "+".join([
                "algorithm", field, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}", f"level{level}"
            ])
            cir = BenchCirData(cir)
            cir_list.append(cir)

        return cir_list

    def _get_benchmark_circuit(self, level: int, q_number: int, ins_set, layout, is_measure):
        cir_list = []
        cirs = BenchmarkCircuitBuilder.get_benchmark_circuit(q_number, level, ins_set, layout)
        for cir in cirs:
            split = cir.name.split("+")
            field, metric = split[1], split[-1]
            cir.inverse() | cir
            if field != "mediate_measure" and is_measure:
                Measure | cir
            cir.name = "+".join([
                "benchmark", field, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}", f"level{level}", metric
            ])
            cir = BenchCirData(cir)
            cir_list.append(cir)

        return cir_list

    def get_circuits(
        self,
        quantum_machine_info: VirtualQuantumMachine,
        level: int = 1,
        enable_qcda_for_alg_cir: bool = False,
        is_measure: bool = False
    ):
        """
        Get circuit from CircuitLib and Get the algorithm circuit after qcda.

        Args:
            quantum_machine_info(VirtualQuantumMachine, optional): The information about the quantum machine.
            level (int): Get the type of benchmark circuit group, include different circuits, one of [1, 2, 3],
                default 1.
            enable_qcda_for_alg_cir(bool): Auto-Compile the circuit with the given quantum machine info, just for
                algorithm circuit, default False.
            is_measure(bool): Can choose whether to measure the circuit according to your needs.

        Returns:
            (List[Circuit]): Return the list of structure for output circuits.
        """

        # obey instruction set and layout in vqm to build random circuit
        q_number = quantum_machine_info.qubit_number
        ins_set = quantum_machine_info.instruction_set
        layout = quantum_machine_info.layout

        # get circuits from circuitlib
        circuit_list = []

        # get random circuits
        circuit_list.extend(self._get_random_circuit(level, q_number, ins_set, layout, is_measure))

        # get benchmark circuits
        circuit_list.extend(self._get_benchmark_circuit(level, q_number, ins_set, layout, is_measure))

        # get algorithm circuit
        circuit_list.extend(
            self._get_algorithm_circuit(quantum_machine_info, level, enable_qcda_for_alg_cir, is_measure)
        )

        return circuit_list

    def run(
        self,
        simulator_interface,
        quantum_machine_info: VirtualQuantumMachine,
        level: int = 1,
        enable_qcda_for_alg_cir: bool = False,
        is_measure: bool = False
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
            is_measure(bool): Can choose whether to measure the circuit according to your needs.

        Returns:
            Return the analysis of benchmarking.
        """
        # Step1 : get circuits from circuitlib
        bench_cir = self.get_circuits(quantum_machine_info, level, enable_qcda_for_alg_cir, is_measure)

        # Step 2: physical machine simulation
        for i in range(len(bench_cir)):
            sim_result = simulator_interface(bench_cir[i].circuit)
            bench_cir[i].machine_amp = sim_result
        self.evaluate(bench_cir)

    def evaluate(self, bench_cir):
        # Step 3: show result
        self.show_result(bench_cir)

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
        feature, values = [], []

        for i in range(len(bench_cir)):
            field = bench_cir[i].field
            gates = bench_cir[i].size
            if field == "random":
                feature.append(f"random{gates}")
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
        ax1.plot(angles_1, values, 'y-', linewidth=2)
        ax1.fill(angles_1, values, 'r', alpha=0.5)

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
                benchmark_score = bench_cir[i].benchmark_score
                value_list.append([field, benchmark_score])
        if len(value_list) > 0:
            field_score_map = defaultdict(list)
            for field, benchmark_score in value_list:
                field_score_map[field].append(benchmark_score)
                feature.append(field)
            feature_2 = list(set(feature))
            feature_2.sort(key=feature.index)
            for value in feature_2:
                values_2.append(max(field_score_map[value]))
            # Sets the angle of the radar chart to bisect a plane
            N = len(values_2)
            alg_data = values_2

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

        ################################ the overall benchmark score #####################################
        data_labels = ('random', 'special', 'algorithm')
        alg_data = list(values_2)
        if len(alg_data) < 4:
            alg_data += (4 - len(alg_data)) * [0]
        else:
            alg_data = random.sample(list(alg_data), 4)
        target_list = [random_data, special_data, alg_data]
        list_1, list_2, list_3, list_4 = [], [], [], []
        for i in range(len(target_list)):
            list_1.append(target_list[i][0])
            list_2.append(target_list[i][1])
            list_3.append(target_list[i][2])
            list_4.append(target_list[i][3])
        values_3 = np.array([list_1, list_2, list_3, list_4])
        angles_3 = np.linspace(0, 2 * np.pi, len(values_3), endpoint=False)
        values_3 = np.concatenate((values_3, [values_3[0]]))
        angles_3 = np.concatenate((angles_3, [angles_3[0]]))

        ax3 = plt.subplot(224, polar=True)
        ax3.plot(angles_3, values_3, 'o-', linewidth=2, alpha=0.2)
        ax3.fill(angles_3, values_3, alpha=0.5)
        ax3.set_ylim(0, np.floor(values_3.max()) + 0.5)
        plt.tick_params(labelsize=12)
        plt.title('The overall benchmark score radar chart show')
        plt.legend(data_labels, loc=(0.94, 0.80), labelspacing=0.1)
        ax3.set_ylim(0, np.floor(values_3.max()) + 4)
        plt.grid(True)

        plt.savefig(self._output_path + "/benchmark_radar_chart_show.jpg")
        plt.show()

    def _txt_show(self, bench_cir):
        result_file = open(self._output_path + '/benchmark_txt_show.txt', mode='w+', encoding='utf-8')
        tb = pt.PrettyTable()
        tb.field_names = [
            'field', 'circuit width', 'circuit size', 'circuit depth', 'I_Score', 'D_Score', 'E_Score', 'benchmark score'
        ]
        for i in range(len(bench_cir)):
            cir = bench_cir[i]
            tb.add_row([cir.field, cir.width, cir.size, cir.depth, cir.I_score, cir.D_score, cir.E_score, cir.benchmark_score])
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
                'I_Score': cir.I_score,
                'D_Score': cir.D_score,
                'E_Score': cir.E_score,
                'benchmark score': cir.benchmark_score
            }
            dfData_list.append(dfData)

        df = pd.DataFrame(dfData_list)
        df.to_excel(self._output_path + "/benchmark_excel_show.xlsx")
