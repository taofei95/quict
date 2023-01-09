import math
import random
import re

import scipy.stats
from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
import numpy as np
import os
from QuICT.tools.circuit_library.circuitlib import CircuitLib

from QuICT.tools.circuit_library.get_benchmark_circuit import BenchmarkCircuitBuilder

based_circuits_list = []
based_fields_list = ["highly_entangled"]
for field in based_fields_list:
    circuits = CircuitLib().get_benchmark_circuit(str(field), qubits_interval=3)
    based_circuits_list.extend(circuits)



from quafu import User
user = User()
user.save_apitoken("EirLbuchxZR12lPmcF2ZNqxWid8PTKNzO6mFwl81SlW.9BzM1gzN2UzN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye")
import numpy as np
from quafu import QuantumCircuit
def quafu(cir):
    qc = QuantumCircuit(cir.width())
    test_cir = cir.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P50", shots=3000, compile=False, priority=2)
    res = task.send(qc)
    
    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 5)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value
    return quafu_amp

amp_results_list = []
for index in range(len(based_circuits_list)):
    amp_results = quafu(based_circuits_list[index])
    amp_results_list.append(amp_results)

def entropy_VQ_score(circuit_list, amp_results_list): 
    def KL_divergence(p, q):
        delta=1e-7
        KL_divergence = 0.5 * scipy.stats.entropy(p+delta, q+delta) + 0.5 * scipy.stats.entropy(q+delta, p+delta)
        return KL_divergence

    def cross_entropy(p, q):
        sum=0.0
        delta=1e-7
        for x in map(lambda y, p:(1 - y) * math.log(1 - p + delta) + y * math.log(p + delta), p, q):
            sum+=x
        cross_entropy = -sum / len(p)
        return cross_entropy

    def L2_loss(p, q):
        delta=1e-7
        L2_loss = np.sum(np.square(p+delta - q+delta))
        return L2_loss


    def normalization(data):
        data = np.array(data)
        data = data/np.sum(data)
        return data
    
    # Step 1: simulate circuit by QuICT simulator
    for index in range(len(circuit_list)):
        sim_result = ConstantStateVectorSimulator().run(circuit_list[index]).get()
        amp = normalization(sim_result)
        amp1 = normalization(amp_results_list[index])
        print(amp, amp1)
        # Step 2: calculate Cross entropy loss, Relative entropy loss, Regression loss
        kl = KL_divergence(np.array(abs(amp)), np.array(abs(amp1)))
        cross_en = cross_entropy(np.array(abs(amp)),np.array(abs(amp1)))
        l2 = L2_loss(np.array(abs(amp)), np.array(abs(amp1)))
        print(kl, cross_en, l2)

    # return kl, cross_en, l2
    
show = entropy_VQ_score(based_circuits_list, amp_results_list)
# print(show)