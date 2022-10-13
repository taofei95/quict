#build env
import json
from quafu import QuantumCircuit
from quafu import User
from QuICT.simulation.unitary.unitary_simulator import UnitarySimulator
user = User()
user.save_apitoken('wRyYinzRHl-VDBRQkMWvi0GcQLpKUQVdMhou2iDtAGL.9JDMwczNxgjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

# qc = QuantumCircuit(4)
# new_path = f"random.qasm"
# with open(new_path) as data_file:
#   data = data_file.read()
#   print(data) 
#   data_content = json.loads(data)
  
# test_cir = """OPENQASM 2.0;
# include "qelib1.inc";
# qreg q[4];
# h q[0];
# h q[1];
# h q[2];
# h q[3];
# """
# qc.from_openqasm(test_cir)
# qc.draw_circuit()

# from quafu import Task
# task = Task()
# task.load_account()
# task.config(backend="ScQ-P10", shots=3000, compile=False)
# res = task.send(qc)

# print(res.counts) #counts
# print(res.amplitudes) #amplitude
# res.plot_amplitudes()

def load_circuit(path):
    from QuICT.tools.interface.qasm_interface import OPENQASMInterface
    #########################################################################
    cir = OPENQASMInterface.load_file(path).circuit

    return cir


def qcda_opt(cir):
    pass


def simu_circuit(cir):
    sim = UnitarySimulator()
    amp1 = sim.run(cir)
    return amp1

def quafu_run(cir):
    qc = QuantumCircuit(5)
    test_cir = cir.qasm()
    qc.from_openqasm(test_cir)

    from quafu import Task
    task = Task()
    task.load_account()
    task.config(backend="ScQ-P10", shots=3000, compile=False, priority=2)
    res = task.send(qc, name="ej")

    quafu_dict = res.amplitudes
    quafu_amp = [0] * (2 ** 5)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value

    return quafu_amp


path = "wr_unit_test/machine-benchmark/randomori.qasm"
cir = load_circuit(path)
amp = simu_circuit(cir)
print(abs(amp))


########################################################################
# cir_opt = OPENQASMInterface.load_file("wr_unit_test/machine-benchmark/randomopt.qasm").circuit
# qc = QuantumCircuit(5)
# test_cir = cir_opt.qasm()
# qc.from_openqasm(test_cir)

# from quafu import Task
# task = Task()
# task.load_account()
# task.config(backend="ScQ-P10", shots=3000, compile=False, priority=2)
# res = task.send(qc, name="ej1")

# quafu_dict2 = res.amplitudes
# quafu_amp2 = [0] * (2 ** 5)
# for key, value in quafu_dict2.items():
#     quafu_amp2[int(key, 2)] = value

# #####################################################################
# sim = UnitarySimulator()
# amp2 = sim.run(cir_opt)
# amp1 = sim.run(cir)

# import numpy as np
# import scipy.stats
# p = np.asarray(abs(amp1))
# q = np.asarray(abs(amp2))

# m = np.asarray(quafu_amp)
# n = np.asarray(quafu_amp2)


# def KL_divergence(p, q):
    # return scipy.stats.entropy(p, q, base=2)
# print((KL_divergence(p, q) + KL_divergence(q, p)) /2)
# print((KL_divergence(p, n) + KL_divergence(n, p)) /2)
# print((KL_divergence(p, m) + KL_divergence(m, p)) /2)
