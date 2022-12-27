import collections
import math
import prettytable as pt
import pandas as pd
import scipy
import numpy as np
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator

from QuICT.tools.circuit_library import CircuitLib
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator


class QuICTBenchmark:
    """ The QuICT Benchmarking. """
    
    __DEFAULT_CLASSIFY = {
        "algorithm": ["adder", "clifford", "grover", "qft", "supremacy", "vqe", "cnf"],
        "benchmark": ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"],
        "random": ["aspen-4", "ourense", "rochester", "sycamore", "tokyo", \
                "ctrl_unitary", "diag", "single_bits", "ctrl_diag", "google", "ibmq", "ionq", "ustc", "nam", "origin"]
    }
    
    def __init__(self, circuit_type: str, output_result_type: str, simulator=ConstantStateVectorSimulator()):
        """
        Initial circuit library

        Args:
            circuit_type (str, optional): one of [circuit, qasm, file]. Defaults to "circuit".
            output_result_type (str, optional): one of [Graph, table, txt file, Excel]. Defaults to "Graph".
            simulator (Union[ConstantStateVectorSimulator, CircuitSimulator], optional): The simulator for simulating quantum circuit. Defaults to CircuitSimulator().
        """
        self._circuit_lib = CircuitLib(circuit_type)
        self._output_type = output_result_type
        self.simulator = simulator

    def get_circuit(
        self,
        fields: List[str],
        max_width: int,
        max_size: int,
        max_depth: int,
        layout_file=False,
        InSet=False
    ):
        """
        Get circuit from CircuitLib and Get the circuit after qcda.

        Args:
            fields (List[str]): The type of circuit required
            max_width(int): max number of qubits
            max_size(int): max number of gates
            max_depth(int): max depth
            layout_file (_type_): Topology of the target physical device
            InSet (List[str]:[one_qubit_gate, [two_qubit_gate]]): The instruction set, only one single-bit gate can be included.

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        cir_alg_list, cir_benchmark_list, cir_random_list, circuit_list = [], [], [], []
        for field in fields:
            if field in self.__DEFAULT_CLASSIFY["algorithm"]:
                cir_alg_list = self._circuit_lib.get_algorithm_circuit(field, max_width, max_size, max_depth)
            elif field in self.__DEFAULT_CLASSIFY["benchmark"]:
                cir_benchmark_list = self._circuit_lib.get_benchmark_circuit(field, max_width, max_size, max_depth)
            elif field in self.__DEFAULT_CLASSIFY["random"]:
                cir_random_list = self._circuit_lib.get_random_circuit(field, max_width, max_size, max_depth)

            circuit_list.extend(cir_alg_list)
            circuit_list.extend(cir_benchmark_list)
            circuit_list.extend(cir_random_list)
            
        # cir_qcda_list = []
        # for circuit in circuit_list:
        #     qcda = QCDA()
        #     if layout_file is not False:
        #         qcda.add_default_mapping(layout_file)
                
        #     if InSet is not False:
        #         q_ins = InstructionSet(InSet[0], InSet[1])
        #         qcda.add_gate_transform(q_ins)
                        
        #     cir_qcda = qcda.compile(circuit)
        #     cir_qcda_list.append(cir_qcda)
            
        return circuit_list
    
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
        print(cir_result_mapping)

        return show
        
    def _field_score(self, field_list: str, cirs_field_map: list, cir_result_mapping):
        # field score
        # Step 1: score each circuit by kl, cross_en, l2 value
        cir_list, based_score_list ,field_score_list = [], [], []
        for i in range(len(field_list)):
            based_score = self._circuit_score(cirs_field_map[field_list[i]])
            based_score_list.extend(based_score)
            for j in range(len(cirs_field_map[field_list[i]])):
                cir_list_index = cirs_field_map[field_list[i]][j]
                cir_list.append(cir_list_index)
        for x in range(len(cir_list)):
            cir = cir_list[x][0]
            # Step 2: get field score from its based_score
            qubit_cal = self._qubit_cal(cir)
            entropy_cal = self._entropy_cal(based_score_list[i][0], based_score_list[i][1], based_score_list[i][2])
            alg_cal = self._alg_cal(cir.name.split("+")[-2])
            field_score_list.append([qubit_cal, entropy_cal, alg_cal])

        # # Step 3: average total field score
        alg_pro_1, alg_pro_2, alg_pro_3 = 0.3, 0.3, 0.4
        rand_pro_1, rand_pro_2 = "50%", "50%"
        alg_proportion = [alg_pro_1, alg_pro_2, alg_pro_3]
        rand_proportion = [rand_pro_1, rand_pro_2]
        
        score_list = []
        for i in range(len(cir_list)):
            if cir.name.split("+")[-2] in self.__DEFAULT_CLASSIFY["algorithm"]:
                result_list = [x*y for x,y in zip(field_score_list[i], alg_proportion)]
            else:
                result_list = [x*y for x,y in zip(field_score_list[i][:-1], rand_proportion)]
            score_list.append(result_list)
        
        return score_list
    
    def _circuit_score(self, circuit_result_group):
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
        
    def _qubit_cal(self, circuit):
        level1, level2, level3 = 60, 80, 100
        if circuit.width() < 10:
            qubit_cal = level1
        if 10 <= circuit.width() < 20:
            qubit_cal = level2
        if 20 <= circuit.width() <= 30:
            qubit_cal = level3
        return qubit_cal
        
    def _entropy_cal(self, kl, cross_en, l2):
        counts = round((kl + cross_en + l2)/3, 6)
        level1, level2, level3 = 60, 80, 100
        if counts <= 0.1:
            counts = level3
        if 0.1 <= counts < 0.2:
            counts = level2
        if 0.2 <= counts <= 0.3:
            counts = level1
        return counts
        
    def _alg_cal(self, field):
        level1, level2, level3 = 60, 80, 100
        if field in self.__DEFAULT_CLASSIFY["algorithm"][0:2]:
            alg_cal = level1
        elif field in self.__DEFAULT_CLASSIFY["algorithm"][2:4]:
            alg_cal = level2
        else:
            alg_cal = level3
        return alg_cal

    def show_result(self, field_list, cir_result_mapping):
        """ show benchmark result. """
        if self._output_type == "Graph-radar":
        # Graph [line, radar, ...]
            show = self._graph_show(field_list, cir_result_mapping, radar=True)
        elif self._output_type == "Graph-scatter":
            show = self._graph_show(field_list, cir_result_mapping, scatter=True)
        # Table
        elif self._output_type == "Table":
            show = self._table_show(cir_result_mapping)
        # txt file
        elif self._output_type == "Txt":
            show = self._txt_show(cir_result_mapping)
        return show

    def _table_show(self, cir_result_mapping):
        print("benchmark table show")
        tb = pt.PrettyTable()
        tb.field_names = ['field', 'circuit width', 'circuit size', 'circuit depth', 'qubit score', 'entropy score', 'alg score', 'Comprehensive score']
        for i in range(len(cir_result_mapping)):
            cirs_field_map = cir_result_mapping[i]
            tb.add_row([cirs_field_map[0], cirs_field_map[1][0].width(), cirs_field_map[1][0].size(), cirs_field_map[1][0].depth(), cirs_field_map[2][0], cirs_field_map[2][1], cirs_field_map[2][2],  cirs_field_map[2][3]])
        
        return tb

    def _graph_show(self, field_list, cir_result_mapping, scatter=False, radar=False):
        if radar == True:
            plt.rcParams['axes.unicode_minus'] = False
            plt.style.use('ggplot')

            #构造数据
            feature = ['circuit width', 'circuit size', 'circuit depth', 'qubit score', 'entropy score', 'alg score']
            # cirs_field_map = dict(cirs_field_map)[field]
            Comprehensive_score_list, max_result_list = [], []
            for i in range(len(cir_result_mapping)):
                Comprehensive_score_list.append(cir_result_mapping[i][2][3])
                min_index = (Comprehensive_score_list.index(min(Comprehensive_score_list)))  
                max_index = (Comprehensive_score_list.index(max(Comprehensive_score_list)))
                min_result = cir_result_mapping[min_index]
                max_result = cir_result_mapping[max_index]
                max_result_list.append(max_result[2][3])
            values = [min_result[1][0].width(), min_result[1][0].size(), min_result[1][0].depth(), min_result[2][0], min_result[2][1], min_result[2][2]]
            values_1 = [max_result[1][0].width(), max_result[1][0].size(), max_result[1][0].depth(), max_result[2][0], max_result[2][1], max_result[2][2]]

            N = len(values)

            #设置雷达图的角度，用于平分切开一个平面
            angles = np.linspace(0,2*np.pi,N,endpoint=False)
            feature = np.concatenate((feature,[feature[0]]))
            values = np.concatenate((values,[values[0]]))
            angles = np.concatenate((angles,[angles[0]]))
            values_1 = np.concatenate((values_1,[values_1[0]]))
            
            #绘图
            plt.figure(figsize=(12,5), dpi=80)
            plt.figure(1)
            ax1 = plt.subplot(121, polar=True)
            ax1.plot(angles,values,'o-',linewidth=2)
            ax1.fill(angles,values,'r',alpha=0.5)
            
            ax1.plot(angles,values_1,'o-',linewidth=2)
            ax1.fill(angles,values_1,'b',alpha=0.5)
            ax1.set_thetagrids(angles*180/np.pi,feature)
            ax1.set_ylim(0,100)
            
            plt.title('Race benchmark radar chart show')
            ax1.grid(True)
            #################################################################
                
            # feature_1 = field_list
            # values_2 = [max_result_list[0], max_result_list[4], max_result_list[7]]
            # feature_1 = ['cnf', 'grover', 'adder', 'ionq']
            # values_2 = [63, 34, 44, 88]

            # N = len(values_2)
            # angles = np.linspace(0,2*np.pi,N,endpoint=False)
            # feature_1 = np.concatenate((feature_1,[feature_1[0]]))
            # values_2 = np.concatenate((values_2,[values_2[0]]))
            # angles = np.concatenate((angles,[angles[0]]))
            # #绘图
            # ax2 = plt.subplot(122, polar=True)
            # ax2.plot(angles,values_2,'o-',linewidth=2)
            # ax2.fill(angles,values_2,'r',alpha=0.5)
            # ax2.set_thetagrids(angles*180/np.pi,feature_1)
            # ax2.set_ylim(0,100)
            
            # plt.title('Comprehensive benchmark radar chart show')
            # ax2.grid(True)
            
            plt.savefig('benchmark radar chart show.jpg')
            plt.show()
            
        if scatter == True:
            pass
            # classify_list, classify_list_new = [], {}
            # # 生成数据
            # x = field_list
            # for i in range(len(cir_result_mapping)):
            #     # cirs_field_map = cir_result_mapping[i]
            #     for j in range(len(field_list)):
            #         if cir_result_mapping[i][0] == field_list[j]:
            #             classify_list = {field_list[j]:cir_result_mapping[i][2][3]}
            #         classify_list_new.update(classify_list)
            #         print(classify_list_new)
            # # y = [a, b, c]
               
            # # 绘图
            # # 1.确定画布
            # plt.figure(figsize=(8, 4))

            # colors = ['black']  # 建立颜色列表
            # labels = ['Comprehensive score value']  # 建立标签类别列表

            # # 2.绘图
            # plt.scatter(x,  # 横坐标
            #             y,  # 纵坐标
            #             c=colors,  # 颜色
            #             label=labels[0])  # 标签

            # # 3.展示图形
            # plt.legend()  # 显示图例

            # plt.show()  # 显示图片
            # plt.savefig('benchmark scatter plot show.jpg') 
            
        #     import matplotlib.pyplot as plt

        #     result = result_dict
        #     y_axis_data = [result_dict["field"], result_dict["circuit_width"], result_dict["circuit_size"], result_dict["circuit_depth"], result_dict["qubit_cal"], result_dict["entropy_cal"], result_dict["alg_cal"], result_dict["Comprehensive_score"]]
        #     x_axis_data = ['field', 'circuit width', 'circuit size', 'circuit depth', 'qubit score', 'entropy score', 'alg score', 'Comprehensive score']

        #     # y_axis_data = [5, 15, 20, 25, 30, 35]
            
        #     for a, b in zip(x_axis_data, y_axis_data):
        #         plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)
                
        #     plt.plot(x_axis_data, y_axis_data, alpha=0.8, linewidth=1)
        #     plt.xlabel('score metrics')
        #     plt.ylabel('score value')

        #     plt.savefig('benchmark line show.jpg') 
        
    def _txt_show(self, cir_result_mapping):
        result_file = open('benchmark txt show.txt','w+')
        tb = pt.PrettyTable()
        tb.field_names = ['field', 'circuit width', 'circuit size', 'circuit depth', 'qubit score', 'entropy score', 'alg score', 'Comprehensive score']
        for i in range(len(cir_result_mapping)):
            cirs_field_map = cir_result_mapping[i]
            tb.add_row([cirs_field_map[0], cirs_field_map[1][0].width(), cirs_field_map[1][0].size(), cirs_field_map[1][0].depth(), cirs_field_map[2][0], cirs_field_map[2][1], cirs_field_map[2][2],  cirs_field_map[2][3]])
        
        result_file.write(str(tb))
        result_file.close()
