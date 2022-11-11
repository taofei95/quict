#build env
from quafu import QuantumCircuit
from quafu import User
from QuICT.qcda.qcda import QCDA

from QuICT.qcda.synthesis.gate_transform.special_set.nam_set import NamSet
user = User()
user.save_apitoken('IzoQuLKdNjviizUAcsw9MUSCya186cqs8ycSVapW21H.9FTM5YTNykjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

#build qasm
import os
import math
from copy import deepcopy
import numpy as np
import scipy.stats
from sklearn import preprocessing as p
import pandas as pd

from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout

layout = Layout.load_file(
    os.path.dirname(os.path.abspath(__file__)) + 
    f"/layout/line5.layout"
)

def load_circuit(path):
    from QuICT.tools.interface.qasm_interface import OPENQASMInterface
    cir = OPENQASMInterface.load_file(path).circuit

    return cir

def simu_cir(cir):
    sim = ConstantStateVectorSimulator()
    amp = sim.run(cir).get()
    return amp

def build_random_circuit_by_topology(circuit: Circuit):
    for i in range(circuit.size()):
        gate = circuit.gates[i]
        if gate.targets + gate.controls == 2:
            a, b = gate.cargs + gate.targs
            if abs(a - b) != 1:
                b = a + 1 if a + 1 != circuit.width() else a - 1
                new_gate = gate & [a, b]
                circuit.replace_gate(i, new_gate)

    return circuit

def ins_auto(circuit):
    qcda = QCDA()
    qcda.add_default_synthesis(NamSet)
    circuit_phy = qcda.compile(circuit)
    cir_opt = AutoOptimization().execute(circuit_phy)
    return cir_opt
    
def ins_qcda(cir):
    qcda_workflow = QCDA()
    qcda_workflow.add_default_synthesis(NamSet)
    qcda_workflow.add_default_optimization()
    # qcda_workflow.add_default_mapping(layout)
    cg_inst = qcda_workflow.compile(cir)

    return cg_inst

def normalization(data):
    data = np.array(data)
    data = data/np.sum(data)

    return data

def KL_divergence(p, q):
    return scipy.stats.entropy(p, q)

def cross_entropy(Y, P):
    sum=0.0
    for x in map(lambda y,p:(1-y)*math.log(1-p)+y*math.log(p),Y,P):
        sum+=x
    return -sum/len(Y)

def L2_loss(y_true,y_pre):
    return np.sum(np.square(y_true-y_pre))

for x in range(1,11):
    path = f"wr_unit_test/machine-benchmark/qasm/quafu/q10_g100_quafu_{x}.qasm"
    circuit = load_circuit(path)
    #function1
    circuit1 = ins_auto(circuit)

    #function2
    circuit2 = ins_qcda(circuit)

    # print(circuit1.qasm())
    # print(circuit2.qasm())
    print(circuit1.size(), circuit2.size(), circuit1.depth(), circuit2.depth())


act_bits_list = []
nd_list = circuit.get_gates_order_by_depth()





# for item in nd_list:
#     act_bits = item.targs + item.cargs
#     act_bits_list.append(act_bits)


# for gate in circuit.gates:
#     act_bit = gate.cargs + gate.targs
#     print(act_bit)

# print(nt)
# p = nd / nt
# circuit.draw()

# for i in range(20):
#     if p > 0.85:
#         print(circuit.qasm())
#         print(p)
#     else:
#         print(f"no")