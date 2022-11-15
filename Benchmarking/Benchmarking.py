import math
import os
from QuICT.core.gate.gate import *
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.lib.circuitlib.circuitlib import CircuitLib
from QuICT.qcda.qcda import QCDA
import scipy.stats
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


class Benchmarking:
    """ The QuICT Benchmarking. """
    
    def __init__(self):
        self._layout_file = None
        self._InSet = None
        
    def get_circuit(self, type: str = None, classify: str = None, max_width: int = None, max_size: int = None, max_depth: int = None):
        """Get algorithm/experiment/random/template circuits in QuICT circuit library."""
        Cirlib = CircuitLib()
        if type == "algorithm":
            cirlib_list = Cirlib.get_algorithm_circuit(classify, max_width, max_size, max_depth)
        if type == "experiment":
            cirlib_list = Cirlib.get_experiment_circuit(classify, max_width, max_size, max_depth)
        if type == "random":
            cirlib_list = Cirlib.get_random_circuit(classify, max_width, max_size, max_depth)
        if type == "template":
            cirlib_list = Cirlib.get_template_circuit(max_width, max_size, max_depth)

        return cirlib_list

    def qcda_circuit(self, circuit_list=list, bench_mapping=False, bench_synthesis=False, bench_optimization=False):
        """ Get the circuit after qcda. """
        for circuit in circuit_list: 
            cir_qcda_list = []

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
            max_list = []
            max_list.append(circuit.width()*circuit.depth())
        return max_list
        
    def Scoring_system(
        self, 
        circuit,
        given_data:list,
        sim_data:list
        ):
   
        KL_divergence = scipy.stats.entropy(sim_data, given_data)

        sum=0.0
        for x in map(lambda y,p:(1-y)*math.log(1-p)+y*math.log(p),sim_data,given_data):
            sum+=x
        cross_entropy = -sum/len(sim_data)

        L2_loss = np.sum(np.square(sim_data-given_data))

        C = (sum(KL_divergence + cross_entropy + L2_loss))/3
        Q = circuit.width()

            



        
