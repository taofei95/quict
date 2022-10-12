#build env
from quafu import QuantumCircuit
from quafu import User
user = User()
user.save_apitoken('wRyYinzRHl-VDBRQkMWvi0GcQLpKUQVdMhou2iDtAGL.9JDMwczNxgjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

#build qasm
import os
from copy import deepcopy
from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.impl_wrapper import MCTSMapping
from QuICT.qcda.mapping.mcts_refactor import MCTSMapping as MCTSMappingRefactor
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.simulation.unitary import *
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout

if __name__=="__main__":
    layout = Layout.load_file(
        os.path.dirname(os.path.abspath(__file__)) + 
        f"/line5.layout"
    )

    single_typelist = [GateType.h, GateType.rx, GateType.ry, GateType.rz] 
    double_typelist = [GateType.cx]
    len_s, len_d = len(single_typelist), len(double_typelist)
    prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

    qubit_num = 5
    cir = Circuit(qubit_num)
    cir.random_append(rand_size=50, typelist=single_typelist + double_typelist, probabilities=prob, random_params=True)

    # mapping
    MCTSMapping = MCTSMappingRefactor
    cir_map = MCTSMapping(layout).execute(deepcopy(cir))
    circuit_map = Circuit(5)
    cir_map | circuit_map
    circuit_map.gate_decomposition()
    #opt
    cir_opt = AutoOptimization().execute(circuit_map)

    # print(circuit_map.qasm())
    # print(cir_opt.qasm())
    # print(circuit_map.depth(),cir_opt.depth())
    
#111111111111111111111111111111111111
    qc = QuantumCircuit(5)
    test_cir = circuit_map.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P10", shots=3000, compile=False, priority=2)
    res = task.send(qc, name="1")

    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 5)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value

#222222222222222222222222222222222222
    qc = QuantumCircuit(5)
    test_cir = cir_opt.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P10", shots=3000, compile=False, priority=2)
    res = task.send(qc, name="11")

    quafu_dict2 = res.amplitudes
    quafu_amp2 = [0] * (2 ** 5)
    for key, value in quafu_dict2.items():
        quafu_amp2[int(key, 2)] = value

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

    sim = UnitarySimulator()
    amp1 = sim.run(circuit_map)
    count_quict = sim.sample(3000)
    # amp2 = sim.run(cir_opt)

    import numpy as np
    import scipy.stats
    p = np.asarray(abs(amp1))
    q = np.asarray(quafu_amp)
    n = np.asarray(quafu_amp2)
    m = np.asarray(quafu_amp3)

    def KL_divergence(p, q):
        return scipy.stats.entropy(p, q, base=2)

    print((KL_divergence(p, q) + KL_divergence(q, p)) /2)
    print((KL_divergence(p, n) + KL_divergence(n, p)) /2)
    print((KL_divergence(p, m) + KL_divergence(m, p)) /2)


