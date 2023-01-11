import os
import random
import re
import pandas as pd
import prettytable as pt
import math
import numpy as np
import scipy.stats
from collections import defaultdict
from matplotlib import pyplot as plt
from QuICT.core.layout.layout import Layout

from QuICT.qcda.qcda import QCDA
from QuICT.tools.circuit_library.circuitlib import CircuitLib
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator

class QuICTBenchmark:
    """ The QuICT Benchmarking. """
    def __init__(self, output_path:str = "./benchmark", show_type:str = "Txt", simulator = ConstantStateVectorSimulator()):
        """
        Initial circuit library

        Args:
            output_path (str, optional): The path of the Analysis of the results.
            show_type (str, optional): Analysis of the Graph exists by default, and other analysis selects Txt or Excel. The default selection is "Txt".
            simulator (Union, optional): The simulator for simulating quantum circuit.
        """
        self._output_path = os.path.abspath(output_path)
        self._show_type = show_type
        self.simulator = simulator

        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)


    def _level_selection(self, qubit_num, level):
        based_circuits_list = []
        based_fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"]
        for field in based_fields_list:
            circuits = CircuitLib().get_benchmark_circuit(str(field), qubits_interval=qubit_num)
            based_circuits_list.extend(circuits)
        
        alg_fields_list = ["aspen-4", "ourense", "rochester", "sycamore", "tokyo", "ctrl_unitary", "diag", "single_bit", "ctrl_diag",
            "google", "ibmq", "ionq", "ustc", "nam", "origin"]
        random_fields_list = random.sample(alg_fields_list, 5)
        for field in random_fields_list:
            circuits = CircuitLib().get_random_circuit(str(field), qubits_interval=qubit_num)
            based_circuits_list.extend(circuits)

        if level >= 2:
            alg_fields_list = ["adder", "clifford", "qft"]
            for field in alg_fields_list:
                circuits = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=qubit_num)
                based_circuits_list.extend(circuits)

        if level == 3:
            alg_fields_list = ["grover", "cnf", "maxcut", "qnn", "quantum_walk", "vqe"]
            for field in alg_fields_list:
                circuits = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=qubit_num)
                based_circuits_list.extend(circuits)

        return based_circuits_list
    
    def get_circuits(self, quantum_machine_info:list, level:int = 1, mapping:bool = False, gate_transform:bool = False):
        """
        Get circuit from CircuitLib and Get the circuit after qcda.

        Args:
            level (int): Get the corresponding level circuits, one of [1, 2, 3]
            quantum_machine_info(list[str]): Gives physical machine properties, for example:[qubits scale, layout_file, InSet]
            layout_file (_type_): Topology of the target physical device
            InSet (List[str]:[one_qubit_gate, [two_qubit_gate]]): The instruction set, only one single-bit gate can be included.

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        #Step 1: get circuits from circuitlib
        circuits_list = self._level_selection(quantum_machine_info[0], level)
        if mapping == False and gate_transform == False:
            return circuits_list

        #Step 2: Whether it goes through QCDA or not
        cir_qcda_list = []
        for circuit in circuits_list:
            qcda = QCDA()
            if mapping == True and len(quantum_machine_info) > 1 and circuit.width() > 1:
                layout_file = quantum_machine_info[1].sub_layout(circuit.width())
                qcda.add_default_mapping(layout_file)
            if gate_transform == True and len(quantum_machine_info) > 2:
                if circuit.name.split("+")[-2] != "mediate_measure":
                    qcda.add_gate_transform(quantum_machine_info[2])

            cir_qcda = qcda.compile(circuit)
            cir_qcda.name = circuit.name
            cir_qcda_list.append(cir_qcda)

        return cir_qcda_list
    
    def run(self, simulator_interface, quantum_machine_info:list, level:int = 1, mapping:bool = False, gate_transform:bool = False):
            """
            Get circuit from CircuitLib and Get the circuit after qcda and get the simulation amplitude results by physical machine.

            Args:
                simulator_interface : Interface for machine machine simulation
                level (int): Get the corresponding level circuits, one of [1, 2, 3]
                quantum_machine_info(list[str]): Gives physical machine properties, for example:[qubits scale, layout_file, InSet]
                layout_file (_type_): Topology of the target physical device
                InSet (List[str]:[one_qubit_gate, [two_qubit_gate]]): The instruction set, only one single-bit gate can be included.

            Returns:
                (List[Circuit | String] | None): Return the list of output circuit order by output_type and the amplitude of the physical machine simulation
            """
            #Step1 : get circuits from circuitlib
            circuits_list = self.get_circuits(quantum_machine_info, level, mapping, gate_transform)
            #Step 2: physical machine simulation
            amp_results_list = []
            for circuit in circuits_list:
                sim_result = simulator_interface(circuit)
                amp_results_list.append(sim_result)
            #Step 3: evaluate all circuits
            self.evaluate(circuits_list, amp_results_list)
            
    def evaluate(self, circuits_list, amp_results_list):
            """
            Evaluate all circuits in circuit list group by fields

            Args:
                circuit_list (List): The list of circuits.
                mac_results_list (List[str]): Physical machine simulation amplitude results
                
            Returns:
                from: Return the benchmark result.
            """
            #Step 1: Entropy measures the difference between the physical machine simulation results and the QUICT simulation results distribution.
            entropy_VQ_score = self._entropy_VQ_score(circuits_list, amp_results_list)

            #Step 2: Filter according to certain conditions to obtain valid circuits.
            valid_circuits_list = self._filter_system(entropy_VQ_score)
            # if valid_circuits_list == []:
            #     print("There is no valid circuit, please select again !")
            #     self.show_result(circuits_list, entropy_score, eigenvalue_VQ_score)

            #Step 3: It is a score for the special benchmark circuit index value and the quantum volume of all circuits.
            eigenvalue_VQ_score = self._eigenvalue_VQ_score(valid_circuits_list)

            #Step 4: Data analysis
            self.show_result(entropy_VQ_score, eigenvalue_VQ_score, valid_circuits_list)

    def _entropy_VQ_score(self, circuit_list, amp_results_list): 
        def normalization(data):
            data = np.array(data)
            data = data/np.sum(data)
            return data
        # Step 1: simulate circuit by QuICT simulator
        entropy_VQ_score = []
        
        for index in range(len(circuit_list)):
            sim_result = self.simulator.run(circuit_list[index]).get()

            quict_result = normalization(abs(sim_result))
            machine_result = normalization(abs(amp_results_list[index]))

            # Step 2: calculate Cross entropy loss, Relative entropy loss, Regression loss
            kl = self._kl_cal(quict_result, machine_result)
            cross_en = self._cross_en_cal(quict_result, machine_result)
            l2 = self._l2_cal(quict_result, machine_result)
            entropy_value = round((abs(kl) + abs(cross_en) + abs(l2))/3, 3)
            entropy_score = self._entropy_cal(entropy_value)
            VQ_value = min(circuit_list[index].width(), circuit_list[index].depth())

            # Step 3: return entropy values and quantum volumn values
            entropy_VQ_score.append([circuit_list[index], entropy_value, entropy_score, VQ_value])
        
        return entropy_VQ_score
            
    def _eigenvalue_VQ_score(self, valid_circuits_list):
        eigenvalue_VQ_score = []
        for i in range(len(valid_circuits_list)):
            field = valid_circuits_list[i].name.split("+")[-2]
            cir_attribute = re.findall(r"\d+", valid_circuits_list[i].name)
            VQ = min(int(cir_attribute[0]), int(cir_attribute[2]))
            if field == "highly_parallelized":
                P = abs((int(cir_attribute[1]) / int(cir_attribute[2]) -1) / (int(cir_attribute[0]) - 1) * VQ)
                eigenvalue_VQ_score.append([valid_circuits_list[i], P])
                
            elif field == "highly_serialized":
                S = (1 - int(cir_attribute[3]) / int(cir_attribute[1])) * VQ
                eigenvalue_VQ_score.append([valid_circuits_list[i], S])
                
            elif field == "highly_entangled":
                E = (1 - int(cir_attribute[3]) / int(cir_attribute[1])) * VQ
                eigenvalue_VQ_score.append([valid_circuits_list[i], E])
                
            elif field == "mediate_measure":
                M = (int(cir_attribute[3]) /  int(cir_attribute[2])) * VQ
                eigenvalue_VQ_score.append([valid_circuits_list[i], M])
        return eigenvalue_VQ_score

    def _kl_cal(self, p, q):
        # calculate KL
        KL_divergence = 0.5 * scipy.stats.entropy(p, q) + 0.5 * scipy.stats.entropy(q, p)
        return KL_divergence
        
    def _cross_en_cal(self, p, q):
        # calculate cross E
        sum=0.0
        delta=1e-7
        for x in map(lambda y, p:(1 - y) * math.log(1 - p + delta) + y * math.log(p + delta), p, q):
            sum+=x
        cross_entropy = -sum / len(p)
        return cross_entropy
            
    def _l2_cal(self, p, q):
        # calculate L2
        delta=1e-7
        L2_loss = np.sum(np.square(p+delta - q+delta))
        return L2_loss

    def _entropy_cal(self, entropy_value):
        counts = round((1 - entropy_value)*100, 1)
        return counts

    def _filter_system(self, entropy_VQ_score):
        valid_circuits_list = []
        for i in range(len(entropy_VQ_score)):
            if entropy_VQ_score[i][2] >= round(2/3 * 100, 3):
                valid_circuits_list.append(entropy_VQ_score[i][0])
        return valid_circuits_list
    
    def show_result(self, entropy_VQ_score, eigenvalue_VQ_score, valid_circuits_list):
        """ show benchmark result. """
        if eigenvalue_VQ_score != []:
            self._graph_show(entropy_VQ_score, eigenvalue_VQ_score, valid_circuits_list)
            if self._show_type == "Txt":
                self._txt_show(entropy_VQ_score)
            else:
                self._excel_show(entropy_VQ_score)
        else:
            if self._show_type == "Txt":
                self._txt_show(entropy_VQ_score)
            else:
                self._excel_show(entropy_VQ_score)

    def _graph_show(self, entropy_VQ_score, eigenvalue_VQ_score, valid_circuits_list):
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('ggplot')
        ################################ based circuits benchmark #####################################
        # Construct the data
        feature = ['parallelized', 'entangled', 'serialized', 'measure', 'VQ']
        P, E, S, M, VQ = [], [], [], [], []
        for i in range(len(eigenvalue_VQ_score)):
            field = eigenvalue_VQ_score[i][0].name.split("+")[-2]
            if field == "highly_parallelized":
                P.append(eigenvalue_VQ_score[i][1])
            elif field == "highly_serialized":
                S.append(eigenvalue_VQ_score[i][1])
            elif field == "highly_entangled":
                E.append(eigenvalue_VQ_score[i][1])
            elif field == "mediate_measure":
                M.append(eigenvalue_VQ_score[i][1])
        alg_fields_list = ["aspen-4", "ourense", "rochester", "sycamore", "tokyo", "ctrl_unitary", "diag", "single_bit", "ctrl_diag",
            "google", "ibmq", "ionq", "ustc", "nam", "origin"]
        for j in range(len(entropy_VQ_score)):
            field = entropy_VQ_score[j][0].name.split("+")[-2]
            if field in alg_fields_list:
                VQ.append(entropy_VQ_score[j][3])
                
        values = [max(P), max(S), max(E), max(M), max(VQ)]
        N = len(values)

        # Sets the angle of the radar chart to bisect a plane
        angles = np.linspace(0,2*np.pi,N,endpoint=False)
        feature = np.concatenate((feature,[feature[0]]))
        values = np.concatenate((values,[values[0]]))
        angles = np.concatenate((angles,[angles[0]]))
        
        # Draw the first diagram
        plt.figure(figsize=(13,5), dpi=80)
        plt.figure(1)
        ax1 = plt.subplot(121, polar=True)
        ax1.plot(angles,values,'y-',linewidth=2)
        ax1.fill(angles,values,'r',alpha=0.5)

        ax1.set_thetagrids(angles*180/np.pi,feature)
        ax1.set_ylim(0,np.floor(values.max())+1)

        plt.title('based circuits benchmark radar chart show')
        ax1.grid(True)
        ################################### algorithmic circuits benchmark ##############################
        # Construct the data
        value_list, feature, values_2 = [], [], []
        for i in range(len(valid_circuits_list)):
            field = valid_circuits_list[i].name.split("+")[-2]
            cir_attribute = re.findall(r"\d+", valid_circuits_list[i].name)
            VQ = min(int(cir_attribute[0]), int(cir_attribute[2]))
            if field == "adder":
                value_list.append([field, VQ])
            if field == "clifford":
                value_list.append([field, VQ])
            if field == "grover":
                value_list.append([field, VQ])
            if field == "qft":
                value_list.append([field, VQ])
            if field == "vqe":
                value_list.append([field, VQ])
            if field == "cnf":
                value_list.append([field, VQ])
            if field == "maxcut":
                value_list.append([field, VQ])
            if field == "qnn":
                value_list.append([field, VQ])
            if field == "quantum_walk":
                value_list.append([field, VQ])

        field_VQ_map = defaultdict(list)
        for field, VQ in value_list:
            field_VQ_map[field].append(VQ)
            feature.append(field)
        feature_1=list(set(feature))
        feature_1.sort(key=feature.index)
        for value in feature_1:
            values_2.append(max(field_VQ_map[value]))
        # Sets the angle of the radar chart to bisect a plane    
        N = len(values_2)
        angles = np.linspace(0,2*np.pi,N,endpoint=False)
        feature_1 = np.concatenate((feature_1,[feature_1[0]]))
        values_2 = np.concatenate((values_2,[values_2[0]]))
        angles = np.concatenate((angles,[angles[0]]))
        # Draw the second diagram
        ax2 = plt.subplot(122, polar=True)
        ax2.plot(angles,values_2,'c-',linewidth=2)
        ax2.fill(angles,values_2,'b',alpha=0.5)
        ax2.set_thetagrids(angles*180/np.pi,feature_1)
        ax2.set_ylim(0, np.floor(values_2.max())+1)

        plt.title('algorithmic circuits benchmark radar chart show')
        ax2.grid(True)

        plt.savefig(self._output_path + "/benchmark_radar_chart_show.jpg")
        plt.show()
    
    def _txt_show(self, entropy_VQ_score):
        result_file = open(self._output_path + '/benchmark_txt_show.txt', mode='w+', encoding='utf-8')
        tb = pt.PrettyTable()
        tb.field_names = ['field', 'circuit width', 'circuit size', 'circuit depth', 'entropy value', 'entropy score', 'VQ value']
        for i in range(len(entropy_VQ_score)):
            field = entropy_VQ_score[i][0].name.split("+")[-2]
            cir_attribute = re.findall(r"\d+", entropy_VQ_score[i][0].name)
            tb.add_row([field, cir_attribute[0], cir_attribute[1], cir_attribute[2], entropy_VQ_score[i][1], entropy_VQ_score[i][2], entropy_VQ_score[i][3]])
        result_file.write(str(tb))
        result_file.close()

    def _excel_show(self, entropy_VQ_score):
        dfData_list = []
        for i in range(len(entropy_VQ_score)):
            field = entropy_VQ_score[i][0].name.split("+")[-2]
            cir_attribute = re.findall(r"\d+", entropy_VQ_score[i][0].name)
            dfData = {
            'field': field,
            'circuit width': cir_attribute[0],
            'circuit size': cir_attribute[1],
            'circuit depth': cir_attribute[2],
            'entropy value': entropy_VQ_score[i][1],
            'entropy score': entropy_VQ_score[i][2],
            'VQ value': entropy_VQ_score[i][3]
            }
            dfData_list.append(dfData)

        df = pd.DataFrame(dfData_list)
        df.to_excel(self._output_path + "/benchmark_excel_show.xlsx")