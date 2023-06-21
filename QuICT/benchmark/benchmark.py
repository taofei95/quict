import os
import re
import pandas as pd
import prettytable as pt
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from QuICT.benchmark.benchlib import BenchLib
from QuICT.core.gate.gate import *

from QuICT.qcda.qcda import QCDA
from QuICT.core import Circuit
from QuICT.tools.circuit_library.circuitlib import CircuitLib


class QuantumMachinebenchmark:
    """ The QuICT Benchmarking. """
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

    def _get_random_circuit(self, level:int, q_number:int, Ins_set, layout=None):
        cir_list = []
        if level == 1:
            gate_prob, pro_s = range(2, 6), 0.9
        elif level == 2:
            gate_prob, pro_s = range(6, 10), 0.8
        elif level == 3:
            gate_prob, pro_s = range(10, 15), 0.7
        for gates in gate_prob:
            cir = Circuit(q_number)
            len_s, len_d = len(Ins_set.one_qubit_gates), len([Ins_set.two_qubit_gate])
            prob = [pro_s / len_s] * len_s + [(1 - pro_s) / len_d] * len_d
            cir.random_append(q_number * gates, Ins_set.gates, probabilities=prob)
            inverse_cgate = cir.to_compositegate().inverse()
            inverse_cgate | cir
            cir.name = "+".join(["random", "random", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}", f"level{level}"])
            cir_list.append(cir)

        return cir_list

    def _get_algorithm_circuit(self, vqm, level:int, enable_qcda_for_alg_cir=False):
        cir_list = []
        alg_fields_list = ["adder", "qft", "clifford", "grover", "cnf", "maxcut", "qnn", "quantum_walk", "vqe"] # clifford can't qcda
        if level == 2:
            field = alg_fields_list[:3]
        if level == 3:
            field = alg_fields_list
        for i in range(len(field)):
            cirs = CircuitLib().get_algorithm_circuit(field[i], qubits_interval=vqm.qubit_number)
            for cir in cirs:
                if enable_qcda_for_alg_cir is True:
                    qcda = QCDA()
                    cir = qcda.auto_compile(cir, vqm)                    
                cir.name = "+".join(["algorithm", field[i], f"w{cir.width()}_s{cir.size()}_d{cir.depth()}", f"level{level}"])
                cir_list.append(cir)

        return cir_list

    def _get_benchmark_circuit(self, level:int, q_number:int):
        cir_list = []
        based_fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"]
        for field in based_fields_list:
            circuits = CircuitLib().get_benchmark_circuit(str(field), qubits_interval=[int(q_number / 2)])
            for cir in circuits:
                inverse_cgate = cir.to_compositegate().inverse()
                inverse_cgate | cir
                cir.name = "+".join(["benchmark", field, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}", f"level{level}"])
                cir_list.append(cir)

        return cir_list

    def get_circuits(
        self,
        quantum_machine_info,
        level: int = 1,
        enable_qcda_for_alg_cir: bool = False
    ):
        """
        Get circuit from CircuitLib and Get the algorithm circuit after qcda.

        Args:
            quantum_machine_info(VirtualQuantumMachine, optional): The information about the quantum machine.
            level (int): Get the type of benchmark circuit group, include different circuits, one of [1, 2, 3],
                default 1.
            enable_qcda_for_alg_cir(bool): Auto-Compile the circuit with the given quantum machine info, just for algorithm circuit,
                default True.

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
        circuit_list.extend(self._get_benchmark_circuit(level, q_number))

        # get algorithm circuit
        circuit_list.extend(self._get_algorithm_circuit(quantum_machine_info, level, enable_qcda_for_alg_cir))

        # list of Special defined Circuit-benchmark data structure
        bench_cir = []
        for i in range(len(circuit_list)):
            benchlib = BenchLib()
            benchlib.circuit = circuit_list[i]
            bench_cir.append(benchlib)

        return bench_cir

    def run(
        self,
        simulator_interface,
        quantum_machine_info,
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

            quantum_machine_info(VirtualQuantumMachine, optional): (VirtualQuantumMachine, optional): The information about the quantum machine.
            level (int): Get the type of benchmark circuit group, include different circuits, one of [1, 2, 3],
                default 1.
            qcda_cir(bool): Auto-Compile the circuit with the given quantum machine info, default True.

        Returns:
            Return the analysis of benchmarking.
        """
        # Step1 : get circuits from circuitlib
        bench_cir = self.get_circuits(quantum_machine_info, level, enable_qcda_for_alg_cir)

        # Step 2: physical machine simulation
        for i in range(len(bench_cir)):
            sim_result = simulator_interface(bench_cir[i].circuit)
            bench_cir[i].machine_amplitude = sim_result

        # Step 2: evaluate each circuit
        # TODO: fidelity or other = self.evaluate(circuit)
        # circuit.fidelity = fidelity

        # TODO: Step 3: show result
        # self.show_result(circuits_list)

    # TODO: use it, self.evaluate(circuits: Union[List[BenchLib], BenchLib]) -> update given circuits, no return
    # def evaluate(self):
    #     """ Evaluate all circuits in circuit list group by fields. """
    #     qv_list, fidelity_list = [], []
    #     for i in range(len(benchlib.field)):
    #         # Quantum volumn
    #         cir_attribute = re.findall(r"\d+", self.benchlib.circuits[i].name)
    #         QV = min(int(cir_attribute[0]), int(cir_attribute[2]))
    #         qv_list.append(QV)
    #         # fidelity
    #         if self.benchlib.field[i] != 'algorithm':
    #             fidelity_list.append(self.benchlib.machine_amp[i][0])
    #         else:
    #             index = self.benchlib.simulation_amp()[i]
    #             fidelity_list.append(self.benchlib.machine_amp[i][index])

    # TODO: self.show_result(circuits)
    # def show_result(self, entropy_QV_score, eigenvalue_QV_score, valid_circuits_list):
    #     """ show benchmark result. """
    #     if not os.path.exists(self._output_path):
    #         os.makedirs(self._output_path)

    #     if len(eigenvalue_QV_score) > 0:
    #         self._graph_show(entropy_QV_score, eigenvalue_QV_score, valid_circuits_list)

    #     if self._output_file_type == "txt":
    #         self._txt_show(entropy_QV_score)

    #     else:
    #         self._excel_show(entropy_QV_score)

    # def _graph_show(self, entropy_QV_score, eigenvalue_QV_score, valid_circuits_list):
    #     plt.rcParams['axes.unicode_minus'] = False
    #     plt.style.use('ggplot')
    #     ################################ based circuits benchmark #####################################
    #     # Construct the data
    #     feature_name = ['parallelized', 'entangled', 'serialized', 'measure', 'QV']

    #     P, E, S, M, QV, values, feature = [], [], [], [], [], [], []
    #     for i in range(len(eigenvalue_QV_score)):
    #         field = eigenvalue_QV_score[i][0].split("+")[-2]
    #         if field == "highly_parallelized":
    #             P.append(eigenvalue_QV_score[i][1])
    #         elif field == "highly_serialized":
    #             S.append(eigenvalue_QV_score[i][1])
    #         elif field == "highly_entangled":
    #             E.append(eigenvalue_QV_score[i][1])
    #         elif field == "mediate_measure":
    #             M.append(eigenvalue_QV_score[i][1])

    #     for j in range(len(entropy_QV_score)):
    #         field_random = entropy_QV_score[j][0].split("+")[-3]
    #         if field_random == "random":
    #             QV.append(float(entropy_QV_score[j][3]))

    #     for x in [P, E, S, M, QV]:
    #         if len(x) > 0:
    #             values.append(max(x))
    #             feature.append(feature_name[[P, E, S, M, QV].index(x)])
    #     N = len(values)

    #     # Sets the angle of the radar chart to bisect a plane
    #     angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    #     feature = np.concatenate((feature, [feature[0]]))
    #     values = np.concatenate((values, [values[0]]))
    #     angles = np.concatenate((angles, [angles[0]]))

    #     # Draw the first diagram
    #     plt.figure(figsize=(13, 5), dpi=100)
    #     plt.figure(1.8)
    #     ax1 = plt.subplot(121, polar=True)
    #     ax1.plot(angles, values, 'y-', linewidth=2)
    #     ax1.fill(angles, values, 'r', alpha=0.5)

    #     ax1.set_thetagrids(angles * 180 / np.pi, feature)
    #     ax1.set_ylim(0, np.floor(values.max()) + 0.5)

    #     plt.tick_params(labelsize=12)
    #     plt.title('based circuits benchmark radar chart show')
    #     ax1.grid(True)

    #     ################################### algorithmic circuits benchmark ##############################
    #     # Construct the data
    #     value_list, feature, values_2 = [], [], []
    #     field_list = ["adder", "clifford", "cnf", "grover", "maxcut", "qft", "qnn", "quantum_walk", "vqe"]
    #     for i in range(len(valid_circuits_list)):
    #         field = valid_circuits_list[i].split("+")[-2]
    #         if field in field_list:
    #             cir_attribute = re.findall(r"\d+", valid_circuits_list[i])
    #             QV = min(int(cir_attribute[0]), int(cir_attribute[2]))
    #             value_list.append([field, QV])
    #     if len(value_list) > 0:
    #         field_QV_map = defaultdict(list)
    #         for field, QV in value_list:
    #             field_QV_map[field].append(QV)
    #             feature.append(field)
    #         feature_1 = list(set(feature))
    #         feature_1.sort(key=feature.index)
    #         for value in feature_1:
    #             values_2.append(max(field_QV_map[value]))
    #         # Sets the angle of the radar chart to bisect a plane
    #         N = len(values_2)
    #         angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    #         feature_1 = np.concatenate((feature_1, [feature_1[0]]))
    #         values_2 = np.concatenate((values_2, [values_2[0]]))
    #         angles = np.concatenate((angles, [angles[0]]))
    #         # Draw the second diagram
    #         ax2 = plt.subplot(122, polar=True)
    #         ax2.plot(angles, values_2, 'c-', linewidth=2)
    #         ax2.fill(angles, values_2, 'b', alpha=0.5)
    #         ax2.set_thetagrids(angles * 180 / np.pi, feature_1)
    #         ax2.set_ylim(0, np.floor(values_2.max()) + 0.5)

    #         plt.tick_params(labelsize=12)
    #         plt.title('algorithmic circuits benchmark radar chart show')
    #         ax2.grid(True)

    #     plt.savefig(self._output_path + "/benchmark_radar_chart_show.jpg")
    #     plt.show()

    # def _txt_show(self, entropy_QV_score):
    #     result_file = open(self._output_path + '/benchmark_txt_show.txt', mode='w+', encoding='utf-8')
    #     tb = pt.PrettyTable()
    #     tb.field_names = [
    #         'field', 'circuit width', 'circuit size', 'circuit depth', 'entropy value', 'entropy score', 'QV value'
    #     ]
    #     for i in range(len(entropy_QV_score)):
    #         field = entropy_QV_score[i][0].split("+")[-2]
    #         cir_attribute = re.findall(r"\d+", entropy_QV_score[i][0])
    #         tb.add_row([
    #             field, cir_attribute[0], cir_attribute[1], cir_attribute[2], entropy_QV_score[i][1],
    #             entropy_QV_score[i][2], (2 ** entropy_QV_score[i][3])
    #         ])
    #     result_file.write(str(tb))
    #     result_file.close()

    # def _excel_show(self, entropy_QV_score):
    #     dfData_list = []
    #     for i in range(len(entropy_QV_score)):
    #         field = entropy_QV_score[i][0].split("+")[-2]
    #         cir_attribute = re.findall(r"\d+", entropy_QV_score[i][0])
    #         dfData = {
    #             'field': field,
    #             'circuit width': cir_attribute[0],
    #             'circuit size': cir_attribute[1],
    #             'circuit depth': cir_attribute[2],
    #             'entropy value': entropy_QV_score[i][1],
    #             'entropy score': entropy_QV_score[i][2],
    #             'QV value': 2 ** entropy_QV_score[i][3]
    #         }
    #         dfData_list.append(dfData)

    #     df = pd.DataFrame(dfData_list)
    #     df.to_excel(self._output_path + "/benchmark_excel_show.xlsx")
