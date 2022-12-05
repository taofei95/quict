import math
import os
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import *
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.tools.circuit_library import CircuitLib
from QuICT.qcda.qcda import QCDA
import scipy.stats
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


class Benchmarking:
    """ The QuICT Benchmarking. """
    
    def __init__(
        self, 
        simulator,
        layout_file: str = None, 
        InSet: str = None, 
        type_list: list =None,
        width: str = None, 
        size : str = None, 
        depth: str = None
        ):
        self.simulator = simulator
        self._layout_file: list = layout_file 
        self._InSet = InSet
        self._type_list = type_list
        self._max_width = width
        self._max_size = size
        self._max_depth = depth
    
    def get_circuit(self):
        """ Get the circuit obey type, width, size, depth from QuICT CircuitLib. """
        for type in self._type_list:
            alg_file = 'QuICT/lib/circuitlib/circuit_qasm/algorithm'
            alg_file_list = os.listdir(alg_file)
            for type_origin in  alg_file_list :
                pass
              
        
        
        
        
        
        
        
        
        
        Cirlib = CircuitLib()
        if self._type == "algorithm":
            circuit_list = Cirlib.get_algorithm_circuit(self._classify, self._max_width, self._max_size, self._max_depth)
        if self._type == "experiment":
            circuit_list = Cirlib.get_experiment_circuit(self._classify, self._max_width, self._max_size, self._max_depth)
        if self._type == "random":
            circuit_list = Cirlib.get_random_circuit(self._classify, self._max_width, self._max_size, self._max_depth)
        if self._type == "template":
            circuit_list = Cirlib.get_template_circuit(self._max_width, self._max_size, self._max_depth)

        return circuit_list
    
    def qcda_circuit(self, circuit_list=list, bench_mapping=False, bench_synthesis=False, bench_optimization=False):
        """ Get the circuit after qcda. """
        cir_qcda_list = []
        for circuit in circuit_list:
            qcda = QCDA()
            if bench_mapping is not False:
                if self._layout_file is not None:
                    layout = self._layout_file
                else:
                    layout = Layout.load_file(
                        os.path.dirname(os.path.abspath(__file__)) + f"/data/layout/line{circuit.width()}.layout"
                        )
                qcda.add_default_mapping(layout)

            if bench_synthesis is not False:
                if self._InSet is not None:
                    q_ins = InstructionSet(self._InSet)
                else:
                    q_ins = InstructionSet(
                            GateType.cx,
                            [GateType.h, GateType.rx, GateType.ry, GateType.rz]
                        )           
                qcda.add_gate_transform(q_ins)
                
            if bench_optimization is not False:
                qcda.add_default_optimization()
                
            cir_qcda = qcda.compile(circuit)
            cir_qcda_list.append(cir_qcda)
       
        return cir_qcda_list

    def filter_circuit(cir_benchmark_list: list):
        for circuit in cir_benchmark_list:
            filter(lambda circuit : max(circuit.width()*circuit.depth()), cir_benchmark_list)
        return circuit
        
    def Scoring_system(
        self, 
        circuit,
        amp_result: list
        ):
        simulator = self.simulator
        sim_result = simulator.run(circuit)
        
        # KL
        KL_divergence = scipy.stats.entropy(sim_result, amp_result)
        #CE
        sum=0.0
        for x in map(lambda y,p:(1-y)*math.log(1-p)+y*math.log(p), sim_result, amp_result):
            sum+=x
        cross_entropy = -sum/len(sim_result)
        #L2
        L2_loss = np.sum(np.square(sim_result - amp_result))

        C = (sum(KL_divergence + cross_entropy + L2_loss))/3
        Q = circuit.width()
        
        return 

            




        
