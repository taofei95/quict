import math
import os
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import *
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
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
        

    def qcda_circuit(self, circuit_list=list, bench_mapping=False, bench_synthesis=False, bench_optimization=False):
        """ Get the circuit after qcda. """
        for circuit in circuit_list:
            cir_origin = circuit.qasm()
            
        cir_benchmark_list = []
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
            qcda.add_default_synthesis(q_ins)
            
        if bench_optimization is not False:
            qcda.add_default_optimization()
            
        cir_benchmark = qcda.compile(cir_origin)
        cir_benchmark_list.append(cir_benchmark)
       
        return cir_benchmark_list

    def filter_circuit(cir_benchmark_list: list):
        for circuit in cir_benchmark_list:
            filter(lambda circuit : max(circuit.width()*circuit.depth()), cir_benchmark_list)
        return circuit
        
    def Scoring_system(
        self, 
        circuit,
        given_data:list,
        sim_data:list,
        KL_divergence = False,
        cross_entropy = False,
        L2_loss = False
        ):
        # sim = ConstantStateVectorSimulator()
        # sim_data = np.array(sim.run(circuit).get())
   
        if KL_divergence is not False:
            KL_divergence = scipy.stats.entropy(sim_data, given_data)

        if cross_entropy is not False:
            sum=0.0
            for x in map(lambda y,p:(1-y)*math.log(1-p)+y*math.log(p),sim_data,given_data):
                sum+=x
            cross_entropy = -sum/len(sim_data)

        if L2_loss is not False:
            L2_loss = np.sum(np.square(sim_data-given_data))

        C = sum(KL_divergence + cross_entropy + L2_loss)
        Q = circuit.width()

            




        
