#build env
from quafu import QuantumCircuit
from quafu import User
user = User()
user.save_apitoken('wRyYinzRHl-VDBRQkMWvi0GcQLpKUQVdMhou2iDtAGL.9JDMwczNxgjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

#build qasm
import os
from copy import deepcopy
import numpy as np
import scipy.stats
from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.impl_wrapper import MCTSMapping
from QuICT.qcda.mapping.mcts_refactor import MCTSMapping as MCTSMappingRefactor
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.simulation.unitary import *
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout

layout = Layout.load_file(
    os.path.dirname(os.path.abspath(__file__)) + 
    f"/line5.layout"
)

single_typelist = [GateType.h, GateType.rx, GateType.ry, GateType.rz] 
double_typelist = [GateType.cx]
len_s, len_d = len(single_typelist), len(double_typelist)
prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

qubit_num = 5
circuit = Circuit(qubit_num)
circuit.random_append(rand_size=50, typelist=single_typelist + double_typelist, probabilities=prob, random_params=True)

def simu_cir(cir):
    sim = UnitarySimulator()
    amp = sim.run(cir)
    return amp

def circuit_opt(cir):
    # mapping
    MCTSMapping = MCTSMappingRefactor
    cir_map = MCTSMapping(layout).execute(deepcopy(cir))
    circuit_map = Circuit(5)
    cir_map | circuit_map
    circuit_map.gate_decomposition()
    #opt
    cir_opt = AutoOptimization().execute(circuit_map)
    return cir_opt
    
def quafu_run(cir, name:str):
    qc = QuantumCircuit(5)
    test_cir = cir.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P10", shots=3000, compile=False, priority=2)
    res = task.send(qc, name)
    counts = res.counts
    
    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 5)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value
    return quafu_amp, counts




run1 = quafu_run(circuit,"circuit")
amp1 = simu_cir(circuit)

cir_opt = circuit_opt(circuit)
run2 = quafu_run(cir_opt,"circuit_opt")
amp2 = simu_cir(cir_opt)

p = np.asarray(abs(amp1))
q = np.asarray(run1)

def KL_divergence(p, q):
    return scipy.stats.entropy(p, q, base=2)

print((KL_divergence(p, q) + KL_divergence(q, p)) /2)
print(abs(amp1))
print(abs(amp2))




















# #222222222222222222222222222222222222
#     qc = QuantumCircuit(5)
#     test_cir = cir_opt.qasm()
#     qc.from_openqasm(test_cir)

#     from quafu import Task
#     task = Task()
#     task.load_account()
#     task.config(backend="ScQ-P10", shots=3000, compile=False, priority=2)
#     res = task.send(qc, name="11")

#     quafu_dict2 = res.amplitudes
#     quafu_amp2 = [0] * (2 ** 5)
#     for key, value in quafu_dict2.items():
#         quafu_amp2[int(key, 2)] = value

#3333333333333333333333333333333333333333
    # qc = QuantumCircuit(5)
    # test_cir = cir.qasm()
    # qc.from_openqasm(test_cir)

    # from quafu import Task
    # task = Task()
    # task.load_account()
    # task.config(backend="ScQ-P10", shots=3000, compile=True, priority=2)
    # res = task.send(qc, name="111")

    # quafu_dict3 = res.amplitudes
    # quafu_amp3 = [0] * (2 ** 5)
    # for key, value in quafu_dict3.items():
    #     quafu_amp3[int(key, 2)] = value

    # sim = UnitarySimulator()
    # amp1 = sim.run(circuit_map)
    # count_quict = sim.sample(3000)
    # # amp2 = sim.run(cir_opt)

    # import numpy as np
    # import scipy.stats
    # p = np.asarray(abs(amp1))
    # q = np.asarray(quafu_amp)
    # n = np.asarray(quafu_amp2)
    # # m = np.asarray(quafu_amp3)

    # def KL_divergence(p, q):
    #     return scipy.stats.entropy(p, q, base=2)

    # print((KL_divergence(p, q) + KL_divergence(q, p)) /2)
    # print((KL_divergence(p, n) + KL_divergence(n, p)) /2)
    # print((KL_divergence(p, m) + KL_divergence(m, p)) /2)


