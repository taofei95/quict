from quafu import QuantumCircuit, User
from quafu import Task

import os
from copy import deepcopy
import numpy as np

import scipy.stats
from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.impl_wrapper import MCTSMapping
from QuICT.qcda.mapping.mcts_refactor import MCTSMapping as MCTSMappingRefactor
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.qcda.qcda import QCDA
from QuICT.simulation.unitary import *
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from QuICT.qcda.synthesis import InstructionSet


user = User()
# user.save_apitoken("AjTW_LbAd717l9rIX8yZloV0hRGuP1-2NH-uY5Foekg.9hTN4EDN4gjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye")
user.save_apitoken('YfeX-7gwHGb3hBcyR8geygJ-8syw2yqNQv68EIvlNOW.9hzM5MDMykjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')
def load_circuit(path):
    from QuICT.tools.interface.qasm_interface import OPENQASMInterface
    cir = OPENQASMInterface.load_file(path).circuit

    return cir

def circuit_trans(cir):
    qcda_workflow = QCDA()
    qf_iset = InstructionSet(
        GateType.cx,
        [GateType.h, GateType.rx, GateType.ry, GateType.rz]
    )
    qcda_workflow.add_default_synthesis(qf_iset)
    cg_inst = qcda_workflow.compile(cir)
    cir_inst = Circuit(5)
    cg_inst | cir_inst
    return cg_inst


def quafu_run(cir, name:str):
    qc = QuantumCircuit(3)
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
        
    return quafu_amp, counts
def simu_cir(cir):
    sim = UnitarySimulator()
    amp = sim.run(cir)
    return amp

path = "wr_unit_test/machine-benchmark/qasm/toffoli.qasm"
cir = load_circuit(path)
cir_g = circuit_trans(cir)
run = quafu_run(cir_g,"toffoli")

amp = simu_cir(cir_g)
print(abs(amp))
