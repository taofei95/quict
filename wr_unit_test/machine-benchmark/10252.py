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
    f"/line5.layout"
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
    
    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 5)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value
    return qc

def quafu_get_cir(cir):
    qc = QuantumCircuit(5)
    test_cir = cir.qasm()
    qc.from_openqasm(test_cir)
    return qc

def quafu_run_trans(qc, name:str):
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
    circuit1 = build_random_circuit_by_topology(circuit)

    cir_ori1 = quafu_get_cir(circuit1)
    cir_ori = quafu_run_trans(cir_ori1,'ori cir')
    amp = normalization(cir_ori)


    cir_opt = circuit_opt(circuit1)
    cir_run1 = quafu_get_cir(cir_opt)
    cir_run = quafu_run_trans(cir_run1,'opt cir')
    amp1 = normalization(cir_run)



    kl = (KL_divergence(abs(amp), abs(amp1)) + KL_divergence(abs(amp1), abs(amp)) /2)
    ce = cross_entropy(abs(amp),abs(amp1))
    l2 = L2_loss(np.array(abs(amp)), np.array(abs(amp1)))

    result_dict = {"circuit_size": cir_ori1.gates,
                #    "circuit_depth": cir_ori.depth(),
                   "cir_opt_size": cir_run1.gates,
                #    "cir_opt_depth": cir_r.depth(),
                   "KL": kl,
                   "CrossEntropy": ce,
                   "L2Loss": l2
                }
    
    return result_dict


df = pd.DataFrame(columns=['RelativeEntropy', 'CrossEntropy', 'L2Loss', 'ori size', 'map size'])
for i in range(1, 11):
    result = qasm_test(i)
    df.loc[i - 1] = [result['KL'], result['CrossEntropy'], result['L2Loss'], result['circuit_size'], result['cir_opt_size']]

print(df)

