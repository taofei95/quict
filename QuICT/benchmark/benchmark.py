

import collections
import math

import numpy as np
import scipy
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
from QuICT.tools.circuit_library.circuitlib import CircuitLib


class QuICTBenchmark:
    """ The QuICT Benchmarking. """
    def __init__(self, circuit_type: str, analysis_type: str, simulator=ConstantStateVectorSimulator()):
        """
        Initial circuit library

        Args:
            circuit_type (str, optional): one of [circuit, qasm, file]. Defaults to "circuit".
            output_result_type (str, optional): one of [Graph, table, txt]. Defaults to "Graph".
            simulator (Union[ConstantStateVectorSimulator, CircuitSimulator], optional): The simulator for simulating quantum circuit. Defaults to CircuitSimulator().
        """
        self._circuit_lib = CircuitLib(circuit_type)
        self._output_type = analysis_type
        self.simulator = simulator
        
    def _level_selection(self, level:str, qubit_num:int):
        based_circuits_list = []
        based_fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"]
        for field in based_fields_list:
                circuits = self._circuit_lib.get_benchmark_circuit(str(field), qubits_interval=qubit_num)
                based_circuits_list.extend(circuits)
        if level == "level1":
            alg_fields_list = ["ionq", "ibmq"]
            for field in based_fields_list:
                circuits = self._circuit_lib.get_random_circuit(str(field), qubits_interval=qubit_num, max_size=qubit_num*5, max_depth=qubit_num*5)
                based_circuits_list.extend(circuits)

        # elif level == "level2":
        #     alg_fields_list = ["adder", "clifford", "qft"]
        #     for field in alg_fields_list:
        #         circuits = self._circuit_lib.get_algorithm_circuit(str(field), qubits_interval=qubit_num, max_size=qubit_num*10, max_depth=qubit_num*10)
        #         based_circuits_list.extend(circuits)

        # elif level == "level3":
        #     alg_fields_list = ["adder", "clifford", "grover", "qft", "vqe", "cnf", "maxcut"]
        #     for field in alg_fields_list:
        #         circuits = self._circuit_lib.get_algorithm_circuit(str(field), qubits_interval=qubit_num, max_size=qubit_num*10, max_depth=qubit_num*10)
        #         based_circuits_list.extend(circuits)

        return based_circuits_list
    
    def get_circuits(self, level:str, quantum_machine_info:list, mapping:bool, synthesis:bool):
        """
        Get circuit from CircuitLib and Get the circuit after qcda.

        Args:
            level (str): Get the corresponding level circuits, one of ["level1", "level2", "level3"]
            quantum_machine_info(list[str]): Gives physical machine properties, for example:[qubits scale, layout_file, InSet]
            layout_file (_type_): Topology of the target physical device
            InSet (List[str]:[one_qubit_gate, [two_qubit_gate]]): The instruction set, only one single-bit gate can be included.

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        #Step1 get circuits from circuitlib
        circuit_list = self._level_selection(level, quantum_machine_info[0])
        #Step2 Whether it goes through QCDA or not
        # cir_qcda_list = []
        # for circuit in circuit_list:
        #     qcda = QCDA()
        #     if mapping is not False:
        #         qcda.add_default_mapping(quantum_machine_info[1])

        #     if synthesis is not False:
        #         q_ins = InstructionSet(quantum_machine_info[2])
        #         qcda.add_gate_transform(q_ins)
                        
        #     cir_qcda = qcda.compile(circuit)
        #     cir_qcda_list.append(cir_qcda)
        
        return circuit_list
    
    def get_circuits_run(self, level, quantum_machine_info:int, mapping:bool, synthesis:bool, simulator):
            """
            Get circuit from CircuitLib and Get the circuit after qcda and get the simulation amplitude results by physical machine.

            Args:
                 level (str): Get the corresponding level circuits
                quantum_machine_info(list[str]): Gives physical machine properties, for example:[qubits scale, layout_file, InSet]
                layout_file (_type_): Topology of the target physical device
                InSet (List[str]:[one_qubit_gate, [two_qubit_gate]]): The instruction set, only one single-bit gate can be included.

            Returns:
                (List[Circuit | String] | None): Return the list of output circuit order by output_type.
            """
            #Step1 get circuits from circuitlib
            circuit_list = self._level_selection(level, quantum_machine_info[0])
            #Step2 Whether it goes through QCDA or not
            cir_qcda_list, results_list = [], []
            for circuit in circuit_list:
                qcda = QCDA()
                if mapping is not False:
                    qcda.add_default_mapping(quantum_machine_info[1])

                elif synthesis is not False:
                    q_ins = InstructionSet(quantum_machine_info[2])
                    qcda.add_gate_transform(q_ins)
                            
                cir_qcda = qcda.compile(circuit)
                cir_qcda_list.append(cir_qcda)
            #Step3 machine simulation
            for circuit in circuit_list:
                simulator = self.simulator
                sim_result = simulator.run(circuit.get())
            results_list.extend(sim_result)

            return circuit_list, results_list
        
    def evaluate(self, circuit_list, result_list):
            """
            Evaluate all circuits in circuit list group by fields

            Args:
                circuit_list (List[str]): the list of circuits group by fields.
                result_list ([dict]): The physical machines simulation result Dict.
                output_type (str, optional): one of [Graph, table, txt file]. Defaults to "Graph".
                circuit_list, result_list must correspond one-to-one.
                
            Returns:
                from: Return the benchmark result.
            """
            # Step 1: Circuits group by fields       
            cir_result_map, cir_result_mapping ,fields_list = [], [], []
            cir_result_map.extend(list(zip(circuit_list, result_list)))
            for i in range(len(cir_result_map)):
                circuit_list = [cir_result_map[i][0].name.split("+")[-2], cir_result_map[i]]
                cir_result_mapping.append(circuit_list)
                # fields_list.append(cir_result_map[i][0].name.split("+")[-2])

            cirs_field_map = collections.defaultdict(list)
            for k, v in cir_result_mapping:
                cirs_field_map[k].append(v)

            # Step 2: Score for each fields in step 1    
            field_list = list(cirs_field_map.keys())
            score_list = self._field_score(field_list, cirs_field_map, cir_result_mapping)

            comprehensive_score = []
            for i in range(len(score_list)):
                all_score = score_list[i][0] + score_list[i][1] + score_list[i][2]
                comprehensive_score.append(all_score)
                score_list[i].append(comprehensive_score[i])
                cir_result_mapping[i].append(score_list[i])

            show = self.show_result(field_list, cir_result_mapping)

            return show
        
    def _field_score(self, field_list: str, cirs_field_map: list, cir_result_mapping):
        # field score
        # Step 1: score each circuit by kl, cross_en, l2 value
        cir_list, based_score_list ,field_score_list = [], [], []
        for i in range(len(field_list)):
            based_score = self._entropy_score(cirs_field_map[field_list[i]])
            based_score_list.extend(based_score)
        #     for j in range(len(cirs_field_map[field_list[i]])):
        #         cir_list_index = cirs_field_map[field_list[i]][j]
        #         cir_list.append(cir_list_index)
        for x in range(len(cir_list)):
            cir = cir_list[x][0]
        # Step 2: Filter according to certain conditions to obtain valid circuits
            valid_circuits = self._filter_system(cir, based_score_list)

        # Step 3: calculate based circuits and algorithmic circuits.
        based_circuits_score = self._metric_property_score()
        algorithmic_circuits_score = self._quantum_volumn_score()
        score_list = []
        
        return score_list
    
    def _entropy_score(self, circuit_result_group):
        from sklearn import preprocessing
        def normalization(data):
            data = np.array(data)
            data = data/np.sum(data)

            return data

        # Step 1: simulate circuit
        circuit_score_list = []
        for i in range(len(circuit_result_group)):
            simulator = self.simulator
            sim_result = normalization(simulator.run(circuit_result_group[i][0]).get())
            mac_result = normalization(circuit_result_group[i][1])

            # Step 2: calculate kl, cross_en, l2, qubit
            kl = self._kl_cal(abs(sim_result), abs(mac_result))
            cross_en = self._cross_en_cal(abs(sim_result), abs(mac_result)) 
            l2 = self._l2_cal(abs(sim_result), abs(mac_result))

            circuit_score_list.append([kl, cross_en, l2])

            # Step 3: return result
        return circuit_score_list
            
    def _metric_property_score(self):
        pass
    def _quantum_volumn_score(self):
        pass
    def _kl_cal(self, p, q):
        # calculate KL
        KL_divergence = 0.5*scipy.stats.entropy(p, q) + 0.5*scipy.stats.entropy(q, p)
        return KL_divergence
        
    def _cross_en_cal(self, p, q):
        # calculate cross E
        sum=0.0
        delta=100
        for x in map(lambda y,p:(1-y)*math.log(1-p+delta) + y*math.log(p+delta), p, q):
            sum+=x
        cross_entropy = -sum/len(p)
        return cross_entropy
            
    def _l2_cal(self, p, q):
        # calculate L2
        delta=1e-7
        L2_loss = np.sum(np.square(p+delta - q+delta))
        return L2_loss

    def _filter_system(self, circuit):
        pass