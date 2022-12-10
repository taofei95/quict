import collections
import math
import scipy
import numpy as np
import pandas as pd
import matplotlib as mpl
from typing import List

from QuICT.tools.circuit_library import CircuitLib
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator


class QuICTBenchmark:
    """ The QuICT Benchmarking. """
    
    __DEFAULT_CLASSIFY = {
        "algorithm": ["adder", "clifford", "grover", "qft", "supremacy", "vqe"],
        "benchmark": ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"],
        "instructionset": ["google", "ibmq", "ionq", "ustc", "quafu"]
    }
    
    def __init__(self, circuit_type: str, output_result_type: str, simulator=CircuitSimulator()):
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
        cir_alg_list, cir_bench_list, cir_ins_list = [], [], []
        for field in fields:
            if field in self.__DEFAULT_CLASSIFY["algorithm"]:
                cir_alg_list = self._circuit_lib.get_algorithm_circuit(field, max_width, max_size, max_depth)
            elif field in self.__DEFAULT_CLASSIFY["benchmark"]:
                cir_bench_list = self._circuit_lib.get_benchmark_circuit(field, max_width, max_size, max_depth)
            else:
                cir_ins_list = self._circuit_lib.get_instructionset_circuit(field, max_width, max_size, max_depth)
                
        circuit_list = cir_alg_list + cir_bench_list + cir_ins_list
        
        cir_qcda_list = []
        # for circuit in circuit_list:
        #     qcda = QCDA()
        #     if layout_file is not False:
        #         qcda.add_default_mapping(layout_file)
                
        #     if InSet is not False:
        #         q_ins = InstructionSet(InSet[0], InSet[1])
        #         qcda.add_gate_transform(q_ins)
                        
        #     cir_qcda = qcda.compile(circuit)
        #     cir_qcda_list.append(cir_qcda)
            
        return circuit_list, cir_qcda_list
    
    def evaluate(self, circuit_list, result_list, output_type=False):
        """
        Evaluate all circuits in circuit list group by fields

        Args:
            circuit_list (List[str]): the list of circuits group by fields.
            result_list ([dict]): The physical machines simulation result Dict.
            output_type (str, optional): one of [Graph, table, txt file]. Defaults to "Graph".
            circuit_id, circuit_list, result_list must correspond one-to-one.
            
        Returns:
            from: Return the benchmark result.
        """
        # Step 1: Circuits group by fields        
        cirs_field_map = collections.defaultdict(list)
        for circuit in circuit_list:
            field = (circuit.name).split('+')[0]    
        cirs_field_map[field].extend(list(zip(circuit_list, result_list)))
       
        # Step 2: Score for each fields in step 1
        score_list  = self._field_score(field, dict(cirs_field_map))
        dict(cirs_field_map)["algorithm"].append(score_list)
        
        # Step 3: Show Result
        result_dict = {"field":field,
                       "circuit_size": circuit.size(),
                    "circuit_width": circuit.width(),
                    "circuit_size": circuit.size(),
                   "circuit_depth": circuit.depth(),
                   "qubit_cal": score_list[0],
                   "entropy_cal": score_list[1],
                   "alg_cal":  score_list[2]
                   }
        
        show = self.show_result(result_dict, output_type)
        
        return show
        
    def _field_score(self, field: str, circuit_result_mapping: dict):
        # field score
        for circuit_result_group in circuit_result_mapping[field]:
        # Step 1: score each circuit by kl, cross_en, l2 value
            based_score = self._circuit_score(circuit_result_group[0], circuit_result_group[1])

        # Step 2: get field score from its based_score
        qubit_cal = self._qubit_cal(circuit_result_group[0])
        entropy_cal = self._entropy_cal(based_score[0], based_score[1], based_score[2])
        alg_cal = self._alg_cal()
        
        score_list = [qubit_cal, entropy_cal, alg_cal]
        
        # # Step 3: average total field score
        alg_pro_1, alg_pro_2, alg_pro_3 = 0.3, 0.3, 0.4
        rand_pro_1, rand_pro_2 = "50%", "50%"
        alg_proportion = [alg_pro_1, alg_pro_2, alg_pro_3]
        rand_proportion = [rand_pro_1, rand_pro_2]
        
        # result_list = []        
        
        if field == "algorithm":
            result_list = [x*y for x,y in zip(score_list, alg_proportion)]
        else:
            result_list = [x*y for x,y in zip(score_list[:-1], rand_proportion)]
            
        return result_list
    
    def _circuit_score(self, circuit, result):
        def normalization(data):
            data = np.array(data)
            data = data/np.sum(data)

            return data
        # Step 1: simulate circuit
        simulator = self.simulator
        sim_result = normalization(simulator.run(circuit))
                
        # Step 2: calculate kl, cross_en, l2, qubit
        mac_result = normalization(result)
        kl = self._kl_cal(abs(sim_result), abs(mac_result))
        cross_en = self._cross_en_cal(abs(sim_result), abs(mac_result))
        l2 = self._l2_cal(abs(sim_result), abs(mac_result))
        
        # Step 3: return result
        return kl, cross_en, l2

    def _kl_cal(self, p, q):
        # calculate KL
        KL_divergence = (scipy.stats.entropy(p, q) + scipy.stats.entropy(q, p)) / 2
        return KL_divergence
        
    def _cross_en_cal(self, p, q):
        # calculate cross E
        sum=0.0
        delta=1e-7
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
        
    def _alg_cal(self):
        return 0

    def show_result(self, result_dict:dict, output_type:str):
        """ show benchmark result. """
        if output_type == "Graph-radar":
        # Graph [line, radar, ...]
            show = self._graph_show(result_dict, radar=True)
        elif output_type == "Graph-line":
            show = self._graph_show(result_dict, line=True)
        # Table
        elif output_type == "Table":
            show = self._table_show(result_dict)
        # txt file
        elif output_type == "Txt":
            show = self._txt_show(result_dict)
        # Excel
        else:
            show = self._excel_show(result_dict)
        return show
            
                
    def _table_show(self, result_dict:dict):
        import prettytable as pt

        tb = pt.PrettyTable()
        tb.field_names = ['field', 'circuit_width', 'circuit_size', 'circuit_depth', 'qubit_cal', 'entropy_cal', 'alg_cal']
        tb.add_row([result_dict["field"], result_dict["circuit_width"], result_dict["circuit_size"], result_dict["circuit_depth"], result_dict["qubit_cal"], result_dict["entropy_cal"], result_dict["alg_cal"]])
        
        return tb
        
    def _graph_show(self, result_dict:dict, line=False, radar=False):
        if radar == True:
            import numpy as np
            import matplotlib.pyplot as plt

            # 中文和负号的正常显示
            plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
            plt.rcParams['axes.unicode_minus'] = False
            plt.style.use('ggplot')

            #构造数据
            result = result_dict
            values = [result["field"], result['circuit_width'], result['circuit_size'], result['qubit_cal'], result['entropy_cal'], result['alg_cal']]
            feature = ['field', 'circuit width', 'circuit size', 'circuit depth', 'qubit cal', 'entropy cal', 'alg cal']

            N = len(values)

            #设置雷达图的角度，用于平分切开一个平面
            angles = np.linspace(0,2*np.pi,N,endpoint=False)
            values = np.concatenate((values,[values[0]]))
            angles = np.concatenate((angles,[angles[0]]))
            #绘图
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles,values,'o-',linewidth=2,label='活动前')
            ax.fill(angles,values,'r',alpha=0.5)
            ax.set_thetagrids(angles*180/np.pi,feature)
            ax.set_ylim(0,5)
            #添加标题
            plt.title('benchmark graph show')
            ax.grid(True)
            
            plt.savefig('benchmark graph show.jpg') 
            
        if line == True:
            import numpy as np
            import matplotlib.pyplot as plt

            result = result_dict
            y_axis_data = [result['circuit_width'], result['circuit_size'], result['circuit_depth'], result['qubit_cal'], result['entropy_cal'], result['alg_cal']]
            x_axis_data = ['circuit width', 'circuit size', 'circuit depth', 'qubit cal', 'entropy cal', 'alg cal']

            # y_axis_data = [5, 15, 20, 25, 30, 35]
            
            for a, b in zip(x_axis_data, y_axis_data):
                plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)
                
            plt.plot(x_axis_data, y_axis_data, alpha=0.8, linewidth=1)
            plt.xlabel('score metrics')
            plt.ylabel('score value')

            plt.savefig('benchmark line show.jpg') 
        
    def _txt_show(self, result_dict:dict):
        result_file = open('benchmark txt show.txt','w+')
        
        df = pd.DataFrame(columns=['circuit width', 'circuit size', 'circuit depth', 'qubit cal', 'entropy cal', 'alg cal', 'field score'])
        result = result_dict
        df = [result['circuit_width'], result['circuit_size'], result['qubit_cal'], result['entropy_cal'], result['alg_cal'], result['field_score']]
        df_1 = "".join(df)
        result_file.write(df_1)
        result_file.close()
    
    def _excel_show(self, result_dict:dict):
        writer = pd.ExcelWriter()
        sheetNames = result_dict.keys()
        data = pd.DataFrame(result_dict)
        for sheetName in sheetNames:
            data.to_excel(writer, sheet_name=sheetName)
        writer.save()
