import math
import os
from imp import reload
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import scipy

from QuICT.core.utils.gate_type import GateType
from QuICT.lib.circuitlib.circuitlib import CircuitLib
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator


class QuICTBenchmark:
    """ The QuICT Benchmarking. """
    
    __DEFAULT_CLASSIFY = {
        "algorithm": ["clifford", "grover", "qft", "supremacy", "vqe"],
        "benchmark": ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"],
        "instructionset": ["google", "ibmq", "ionq", "ustc", "quafu"],
    }
    
    def __init__(self, circuit_type: str, output_result_type: str, simulator=CircuitSimulator()):
        """
        Initial circuit library

        Args:
            circuit_type (str, optional): one of [circuit, qasm, file]. Defaults to "circuit".
            output_result_type (str, optional): one of [Graph, table, txt file]. Defaults to "Graph".
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
        max_depth: int
    ):
        """
        Get circuit from CircuitLib

        Args:
            fields (List[str]): The type of circuit required
            max_width(int): max number of qubits
            max_size(int): max number of gates
            max_depth(int): max depth

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        circuit_list = []
        for field in fields:
            if field in self.__DEFAULT_CLASSIFY["algorithm"]:
                cir_alg_list = self._circuit_lib.get_algorithm_circuit(field, max_width, max_size, max_depth)
            elif field in self.__DEFAULT_CLASSIFY["benchmark"]:
                cir_bench_list = self._circuit_lib.get_benchmark_circuit(field, max_width, max_size, max_depth)
            else:
                cir_ins_list = self._circuit_lib.get_instructionset_circuit(field, max_width, max_size, max_depth)
                
            circuit_list.append(cir_alg_list)
            
        return circuit_list
    
    # def qcda_circuit(self, layout_file, InSet, circuit_list=list, bench_mapping=False, bench_synthesis=False, bench_optimization=False):
    #     """
    #     Get the circuit after qcda.

    #     Args:
    #         layout_file (_type_): Topology of the target physical device
    #         InSet (List[str]): The instruction set, only one single-bit gate can be included.
    #         circuit_list (List[str]): the list of circuits. Defaults to list.
    #         bench_mapping (bool, optional): select whether to mapping. Defaults to False.
    #         bench_synthesis (bool, optional): select whether to gate transform. Defaults to False.
    #         bench_optimization (bool, optional): select whether to optimization. Defaults to False.

    #     Returns:
    #         (List[Circuit | String] | None): Return the list of circuits after qcda.
    #     """
    #     cir_qcda_list = []
    #     for circuit in circuit_list:
    #         qcda = QCDA()
    #         if bench_mapping is not False:
    #             qcda.add_default_mapping(layout_file)

    #         if bench_synthesis is not False:
    #             if InSet is not None:
    #                 q_ins = InstructionSet(InSet)
    #             else:
    #                 q_ins = InstructionSet(
    #                         GateType.cx,
    #                         [GateType.h, GateType.rx, GateType.ry, GateType.rz]
    #                     )           
    #             qcda.add_gate_transform(q_ins)
                
    #         if bench_optimization is not False:
    #             qcda.add_default_optimization()
                
    #         cir_qcda = qcda.compile(circuit)
    #         cir_qcda_list.append(cir_qcda)
       
    #     return cir_qcda_list
    
    # def evaluate(self, circuit_list, result_list, output_type):
    #     """
    #     Evaluate all circuits in circuit list group by fields

    #     Args:
    #         circuit_id (List[str]): the list of the unique id for circuits, for example : w2_s12_d10.
    #         circuit_list (List[str]): the list of circuits group by fields.
    #         result_list ([dict]): The physical machines simulation result Dict.
    #         output_type (str, optional): one of [Graph, table, txt file]. Defaults to "Graph".
    #         circuit_id, circuit_list, result_list must correspond one-to-one.
            
    #     Returns:
    #         from: Return the benchmark result.
    #     """
    #     # Step 1: Circuits group by fields
    #     for circuit in circuit_list:
    #         for result in result_list:
    #             cirs_field_mapping = {f"{field}": [(circuit, result)]}

    #     # Step 2: Score for each fields in step 1
    #     for field in cirs_field_mapping:
    #         self._field_score(field, cirs_field_mapping[f"{field}"])

    #     # Step 3: Show Result
    #     self.show_result(output_type)
        
    # def _field_score(self, field: str, circuit_result_mapping: List[Tuple]):
    #     # field score
    #     for circuit, result in circuit_result_mapping[f"{field}"]:
    #     # Step 1: score each circuit by kl, cross_en, l2 value
    #         based_score = self._circuit_score(circuit, result)[:-1]

    #     # Step 2: get field score from its based_score
                
    #     # Step 3: average total field score
    #     alg_pro_1, alg_pro_2, alg_pro_3 = "30%", "30%", "40%"
    #     rand_pro_1, rand_pro_2 = "50%", "50%"
    #     alg_proportion = [alg_pro_1, alg_pro_2, alg_pro_3]
    #     rand_proportion = [rand_pro_1, rand_pro_2]
        
    #     if field in self.__DEFAULT_CLASSIFY["algorithm"]:
    #         result_list = [x*y for x,y in zip(self._circuit_score(circuit, result), alg_proportion)]
            
    #     result_list = [x*y for x,y in zip(self._circuit_score(circuit, result)[:-1], rand_proportion)]
            
    #     return result_list
    
    # def _circuit_score(self, circuit, result):
    #     # Step 1: simulate circuit
    #     simulator = self.simulator
    #     sim_result = simulator.run(circuit)
        
    #     # Step 2: calculate kl, cross_en, l2, qubit
    #     mac_result = result
    #     kl = self._kl_cal(sim_result, mac_result)
    #     cross_en = self._cross_en_cal(sim_result, mac_result)
    #     l2 = self._l2_cal(sim_result, mac_result)

    #     # Step 3: calculate level
    #     qubit_cal = self._qubit_cal(circuit)
    #     entropy_cal = self._entropy_cal(kl, cross_en, l2)
    #     alg_cal = self._alg_cal()
        
    #     score_list = [qubit_cal, entropy_cal, alg_cal]
        
    #     # Step 3: return result
    #     return score_list

    # def _kl_cal(self, p, q):
    #     # calculate KL
    #     KL_divergence = scipy.stats.entropy(p, q)
    #     return KL_divergence
        
    # def _cross_en_cal(self, p, q):
    #     # calculate cross E
    #     sum=0.0
    #     for x in map(lambda y,p:(1-y)*math.log(1-p)+y*math.log(p), p, q):
    #         sum+=x
    #     cross_entropy = -sum/len(p)
    #     return cross_entropy
            
    # def _l2_cal(self, p, q):
    #     # calculate L2
    #     L2_loss = np.sum(np.square(p - q))
    #     return L2_loss
        
    # def _qubit_cal(self, circuit):
    #     level1, level2, level3 = 60, 80, 100
    #     if circuit.width() < 10:
    #         qubit_cal = level1
    #     if 10 <= circuit.width() < 20:
    #         qubit_cal = level2
    #     if 20 <= circuit.width() <= 30:
    #         qubit_cal = level3
    #     return qubit_cal
        
    # def _entropy_cal(self, kl, cross_en, l2):
    #     counts = (round(kl + cross_en + l2)/3, 6)
    #     level1, level2, level3 = 60, 80, 100
    #     if counts <= 0.1:
    #         counts = level3
    #     if 0.1 <= counts < 0.2:
    #         counts = level2
    #     if 0.2 <= counts <= 0.3:
    #         counts = level1
    #     return counts
        
    # def _alg_cal(self):
    #     pass

    # def show_result(self, output_type:str):
    #     """ show benchmark result. """
    #     if output_type == "Graph":
    #     # Graph [line, radar, ...]
    #         Graph = self._graph_show()
    #     # Table
    #     if output_type == "Table":
    #         Table = self._table_show()
    #     # txt file
    #     if output_type == "Txt":
    #         Txt = self._txt_show()
                
    # def _table_show(self):
    #     import sys
    #     from prettytable import PrettyTable
    #     reload(sys)
    #     sys.setdefaultencoding('utf8')

    #     table = PrettyTable()
    #     table.add_column('项目', ['编号','云编号','名称','IP地址'])
    #     table.add_column('值', ['1','server01','服务器01','172.16.0.1'])
    #     return table
        
    # def _graph_show(self, line=False, radar=False):
    #     results = [{"大学英语": 87, "高等数学": 79, "体育": 95, "计算机基础": 92, "程序设计": 85},
    #             {"大学英语": 80, "高等数学": 90, "体育": 91, "计算机基础": 85, "程序设计": 88}]
    #     data_length = len(results[0])
    #     # 将极坐标根据数据长度进行等分
    #     angles = np.linspace(0, 2*np.pi, data_length, endpoint=False)
    #     labels = [key for key in results[0].keys()]
    #     score = [[v for v in result.values()] for result in results]
    #     # 使雷达图数据封闭
    #     score_a = np.concatenate((score[0], [score[0][0]]))
    #     score_b = np.concatenate((score[1], [score[1][0]]))
    #     angles = np.concatenate((angles, [angles[0]]))
    #     labels = np.concatenate((labels, [labels[0]]))
    #     # 设置图形的大小
    #     fig = plt.figure(figsize=(8, 6), dpi=100)
    #     # 新建一个子图
    #     ax = plt.subplot(111, polar=True)
    #     # 绘制雷达图
    #     ax.plot(angles, score_a, color='g')
    #     ax.plot(angles, score_b, color='b')
    #     # 设置雷达图中每一项的标签显示
    #     ax.set_thetagrids(angles*180/np.pi, labels)
    #     # 设置雷达图的0度起始位置
    #     ax.set_theta_zero_location('N')
    #     # 设置雷达图的坐标刻度范围
    #     ax.set_rlim(0, 100)
    #     # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
    #     ax.set_rlabel_position(270)
    #     ax.set_title("QuICT benchmark Graph")
    #     plt.legend(["弓长张", "口天吴"], loc='best')
    #     plt.show()
        
    # def _txt_show(self):
    #     result_file = open('recode.txt', mode = 'a',encoding='utf-8')
    #     print({}, file=result_file)
    #     result_file.close()