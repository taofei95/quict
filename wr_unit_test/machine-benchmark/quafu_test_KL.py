#build env
from quafu import QuantumCircuit
from quafu import User
user = User()
user.save_apitoken('IzoQuLKdNjviizUAcsw9MUSCya186cqs8ycSVapW21H.9FTM5YTNykjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

#build qasm
import os
from copy import deepcopy
import numpy as np
import scipy.stats
from sklearn import preprocessing as p
import pandas as pd

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

for i in range(1, 11):
    path = f"wr_unit_test/machine-benchmark/qasm/quafu/q10_g100_quafu_{i}.qasm"
    id = 6
    circuit = load_circuit(path)
    
    cir_map = build_random_circuit_by_topology(circuit)

    amp = normalization(simu_cir(cir_map))

    amp1 = normalization(quafu_run(cir_map,f"circuit_map_KL_{id}"))

    cir_opt = circuit_opt(cir_map)
    amp2 = normalization(quafu_run(cir_opt,f"circuit_opt_KL_{id}"))

    cir_opt_trans = circuit_opt(circuit)
    amp3 = normalization(quafu_run_trans(cir_opt_trans,f"circuit_trans_KL_{id}"))

    def KL_divergence(p, q):
        return scipy.stats.entropy(p, q)

    print(f"KL")
    print((KL_divergence(abs(amp), abs(amp1)) + KL_divergence(abs(amp1), abs(amp))) /2)
    print((KL_divergence(abs(amp), abs(amp2)) + KL_divergence(abs(amp2), abs(amp))) /2)
    print((KL_divergence(abs(amp), abs(amp3)) + KL_divergence(abs(amp3), abs(amp))) /2)

    import math
    def cross_entropy(Y, P):
        sum=0.0
        for x in map(lambda y,p:(1-y)*math.log(1-p)+y*math.log(p),Y,P):
            sum+=x
        return -sum/len(Y)

    print(f"ce")
    print(cross_entropy(abs(amp),abs(amp1)))
    print(cross_entropy(abs(amp),abs(amp2)))
    print(cross_entropy(abs(amp),abs(amp3)))

    def L2_loss(y_true,y_pre):
            return np.sum(np.square(y_true-y_pre))

        #假设我们得到了真实值和预测值
        #定义函数

    print(f"L2")
    print(L2_loss(np.array(abs(amp)), np.array(abs(amp1))))
    print(L2_loss(np.array(abs(amp)), np.array(abs(amp2))))
    print(L2_loss(np.array(abs(amp)), np.array(abs(amp3))))

    print(f"opt size:{circuit.size()}, opttrans size:{cir_opt.size()}, opt depth:{circuit.depth()} opttrans depth:{cir_opt.depth()}")
    # data = [(KL_divergence(abs(amp), abs(amp1)) + KL_divergence(abs(amp1), abs(amp)) /2), cross_entropy(abs(amp),abs(amp1)), L2_loss(np.array(abs(amp)), np.array(abs(amp1))), circuit.size(), cir_opt.size(), circuit.depth(), cir_opt.depth()]
    

# for j in range(1):
#     data_list = []
#     data_list.append(data)
#     df = pd.DataFrame(data_list)
#     df.to_excel("cross entropy.xlsx",index=False)


















# file1 = ([0.089, 0.08733333333333333, 0.03933333333333333, 0.034666666666666665, 0.04766666666666667, 0.05366666666666667, 0.15933333333333333, 0.09566666666666666, 0.027666666666666666, 0.01633333333333333, 0.016, 0.004666666666666667, 0.011333333333333334, 0.021666666666666667, 0.011333333333333334, 0.015, 0.02, 0.017333333333333333, 0.014333333333333333, 0.011333333333333334, 0.020666666666666667, 0.020666666666666667, 0.03666666666666667, 0.033666666666666664, 0.013666666666666667, 0.009666666666666667, 0.011666666666666667, 0.010666666666666666, 0.012333333333333333, 0.014, 0.008666666666666666, 0.014])
# file2 = ([0.11566666666666667, 0.10666666666666667, 0.03833333333333333, 0.056, 0.04933333333333333, 0.033666666666666664, 0.267, 0.11633333333333333, 0.036, 0.03, 0.016666666666666666, 0.012666666666666666, 0.018333333333333333, 0.02266666666666667, 0.02266666666666667, 0.024333333333333332, 0.0016666666666666668, 0.004333333333333333, 0.0023333333333333335, 0.0006666666666666666, 0.0023333333333333335, 0.0026666666666666666, 0.005666666666666667, 0.0026666666666666666, 0.0026666666666666666, 0.002, 0.002, 1e-9, 0.0013333333333333333, 0.0003333333333333333, 0.0023333333333333335, 0.0006666666666666666])
# file3 = ([0.13933333333333334, 0.16633333333333333, 0.048, 0.04566666666666667, 0.06766666666666667, 0.050333333333333334, 0.13633333333333333, 0.07733333333333334, 0.047, 0.027666666666666666, 0.03266666666666666, 0.016666666666666666, 0.014666666666666666, 0.014666666666666666, 0.04633333333333333, 0.046, 0.004666666666666667, 0.003, 0.0013333333333333333, 0.0003333333333333333, 0.0023333333333333335, 0.0023333333333333335, 1e-9, 0.0026666666666666666, 0.0013333333333333333, 1e-9, 0.0003333333333333333, 1e-9, 1e-9, 1e-9, 0.002, 0.003])
# file4 = ([0.17633333333333334, 0.113, 0.030666666666666665, 0.06133333333333333, 0.02666666666666667, 0.060333333333333336, 0.16466666666666666, 0.10166666666666667, 0.051, 0.056666666666666664, 0.024666666666666667, 0.017666666666666667, 0.023666666666666666, 0.031, 0.027, 0.013333333333333334, 0.0023333333333333335, 0.0006666666666666666, 0.0013333333333333333, 0.0016666666666666668, 0.0006666666666666666, 0.0013333333333333333, 0.0003333333333333333, 0.004666666666666667, 0.001, 0.0013333333333333333, 1e-9, 0.0013333333333333333, 0.003, 1e-9, 1e-9, 0.0006666666666666666])

# Q = np.asarray(abs(amp))
# p = np.asarray(file1)
# q = np.asarray(file2)
# m = np.asarray(file3)
# n = np.asarray(file4)

# def KL_divergence(p, q):
#     return scipy.stats.entropy(p, q, base=2)

# print((KL_divergence(p, Q) + KL_divergence(Q, p)) /2)
# print((KL_divergence(Q, q) + KL_divergence(q, Q)) /2)
# print((KL_divergence(m, Q) + KL_divergence(Q, m)) /2)
# print((KL_divergence(n, Q) + KL_divergence(Q, n)) /2)


# print(abs(amp1))
# print(abs(amp2))


