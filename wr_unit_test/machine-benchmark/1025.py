#build env
from quafu import QuantumCircuit
from quafu import User
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
    f"/line10.layout"
)

# single_typelist = [GateType.h, GateType.rx, GateType.ry, GateType.rz] 
# double_typelist = [GateType.cx]
# len_s, len_d = len(single_typelist), len(double_typelist)
# prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

# qubit_num = 5
# circuit = Circuit(qubit_num)
# circuit.random_append(rand_size=10, typelist=single_typelist + double_typelist, probabilities=prob, random_params=True)

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
            
def circuit_opt(cir):
    cir_opt = AutoOptimization().execute(cir)
    return cir_opt
    
def quafu_run(cir, name:str):
    qc = QuantumCircuit(5)
    test_cir = cir.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P20", shots=3000, compile=False, priority=2)
    res = task.send(qc, name)
    counts = res.counts
    
    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 5)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value
    return quafu_amp

def quafu_run_trans(cir, name:str):
    qc = QuantumCircuit(5)
    test_cir = cir.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P20", shots=3000, compile=True, priority=2)
    res = task.send(qc, name)
    counts = res.counts
    
    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 5)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value
    return quafu_amp

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

def qasm_test(x):
    path = f"wr_unit_test/machine-benchmark/qasm/quafu/q5_g50_quafu_{x}.qasm"
    circuit = load_circuit(path)

    # amp = normalization(simu_cir(circuit))

    cir_1 = build_random_circuit_by_topology(circuit)
    amp = normalization(quafu_run(cir_1,'ori cir'))

    cir_opt = circuit_opt(cir_1)
    amp1 = normalization(quafu_run(cir_opt,'map cir'))

    kl = (KL_divergence(abs(amp), abs(amp1)) + KL_divergence(abs(amp1), abs(amp)) /2)
    ce = cross_entropy(abs(amp),abs(amp1))
    l2 = L2_loss(np.array(abs(amp)), np.array(abs(amp1)))
    result_dict = {"circuit_size": circuit.size(),
                   "circuit_depth": circuit.depth(),
                   "cir_opt_size": cir_1.size(),
                   "cir_opt_depth": cir_1.depth(),
                   "KL": kl,
                   "CrossEntropy": ce,
                   "L2Loss": l2}
    
    return result_dict


df = pd.DataFrame(columns=['RelativeEntropy', 'CrossEntropy', 'L2Loss', 'ori size', 'map size', 'ori depth', ',map depth'])
for i in range(1, 11):
    result = qasm_test(i)
    df.loc[i - 1] = [result['KL'], result['CrossEntropy'], result['L2Loss'], result['circuit_size'], result['cir_opt_size'], result['circuit_depth'], result['cir_opt_depth']]

print(df)

