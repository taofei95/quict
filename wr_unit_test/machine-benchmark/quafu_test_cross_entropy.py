#build env
import pandas as pd
from quafu import QuantumCircuit
from quafu import User
user = User()
user.save_apitoken('IzoQuLKdNjviizUAcsw9MUSCya186cqs8ycSVapW21H.9FTM5YTNykjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

#build qasm
import os
from copy import deepcopy
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.impl_wrapper import MCTSMapping
from QuICT.qcda.mapping.mcts_refactor import MCTSMapping as MCTSMappingRefactor
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout

layout = Layout.load_file(
    os.path.dirname(os.path.abspath(__file__)) + 
    f"/line10.layout"
)

single_typelist = [GateType.h, GateType.rx, GateType.ry, GateType.rz] 
double_typelist = [GateType.cx]
len_s, len_d = len(single_typelist), len(double_typelist)
prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

qubit_num = 10
circuit = Circuit(qubit_num)
circuit.random_append(rand_size=100, typelist=single_typelist + double_typelist, probabilities=prob, random_params=True)

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

# def circuit_map(cir):
#     # mapping
#     MCTSMapping = MCTSMappingRefactor
#     cir_map = MCTSMapping(layout).execute(deepcopy(cir))
#     circuit_map = Circuit(10)
#     cir_map | circuit_map
#     circuit_map.gate_decomposition()
#     return cir_map
    
def circuit_opt(cir):
    # mapping
    MCTSMapping = MCTSMappingRefactor
    cir_map = MCTSMapping(layout).execute(deepcopy(cir))
    circuit_map = Circuit(10)
    cir_map | circuit_map
    circuit_map.gate_decomposition()
    #opt
    cir_opt = AutoOptimization().execute(circuit_map)
    return cir_opt
    
def quafu_run(cir, name:str):
    qc = QuantumCircuit(10)
    test_cir = cir.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P20", shots=3000, compile=False, priority=2)
    res = task.send(qc, name)
    counts = res.counts
    
    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 10)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value
    return quafu_amp

def quafu_run_trans(cir, name:str):
    qc = QuantumCircuit(10)
    test_cir = cir.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P20", shots=3000, compile=True, priority=2)
    res = task.send(qc, name)
    counts = res.counts
    
    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 10)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value
    return quafu_amp

def normalization(data):
    data = np.array(data)
    data = data/np.sum(data)

    return data



amp = normalization(abs(simu_cir(circuit)))

cir_map = build_random_circuit_by_topology(circuit)
amp1 = normalization(quafu_run(cir_map,"circuit_map"))


cir_opt = circuit_opt(circuit)
amp2 = normalization(quafu_run(cir_opt,"circuit_opt"))

amp3 = normalization(quafu_run_trans(circuit,"circuit_trans"))

import math
def cross_entropy(Y, P):
    sum=0.0
    for x in map(lambda y,p:(1-y)*math.log(1-p)+y*math.log(p),Y,P):
        sum+=x
    return -sum/len(Y)
data = [cross_entropy(abs(amp),abs(amp1)), cross_entropy(abs(amp),abs(amp2)), cross_entropy(abs(amp),abs(amp3))]


print(data)

# for j in range(3):
#     data_list = []
#     data_list.append(data)
#     df = pd.DataFrame(data_list)
#     df.to_excel("cross entropy.xlsx",index=False)
    
