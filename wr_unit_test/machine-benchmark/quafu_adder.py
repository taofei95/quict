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
# from QuICT.qcda.synthesis.gate_decomposition.gate_decomposition import GateDecomposition
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from QuICT.qcda.synthesis import InstructionSet


user = User()
user.save_apitoken('whGwcN-inXsj_KWZfhFqCNyIUvCRkQt3KUx2htaVy3h.9ljN0cTMwkjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')
layout = Layout.load_file(
    os.path.dirname(os.path.abspath(__file__)) + 
    f"/line3.layout"
)

def load_circuit(path):
    from QuICT.tools.interface.qasm_interface import OPENQASMInterface
    cir = OPENQASMInterface.load_file(path).circuit

    return cir

def circuit_trans(cir, qubits):
    qcda_workflow = QCDA()
    qf_iset = InstructionSet(
        GateType.cx,
        [GateType.h, GateType.rx, GateType.ry, GateType.rz]
    )
    qcda_workflow.add_default_synthesis(qf_iset)
    # qcda_workflow.add_default_optimization()
    # qcda_workflow.add_default_mapping(layout)
    cg_inst = qcda_workflow.compile(cir)
    cir_inst = Circuit(qubits)
    cg_inst | cir_inst
    return cg_inst

def circuit_opt(cir):
    # mapping
    MCTSMapping = MCTSMappingRefactor
    cir_map = MCTSMapping(layout).execute(deepcopy(cir))
    circuit_map = Circuit(3)
    cir_map | circuit_map
    circuit_map.gate_decomposition()
    #opt
    cir_opt = AutoOptimization().execute(circuit_map)
    return cir_opt

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
    quafu_amp = [0] * (2 ** 3)
    for key, value in quafu_dict.items():
        quafu_amp[int(key, 2)] = value
        
    return quafu_amp
def simu_cir(cir):
    sim = ConstantStateVectorSimulator()
    amp = sim.run(cir).get()
    return amp

def opt(qf_iset):
    qf_iset = InstructionSet(
        GateType.cx,
        [GateType.h, GateType.rx, GateType.ry, GateType.rz]
    )
    qcda_workflow = QCDA()
    qcda_workflow.add_default_synthesis(qf_iset)
    qcda_workflow.add_default_optimization()
    qcda_workflow.add_default_mapping(layout)
    return qcda_workflow

path = "wr_unit_test/machine-benchmark/plus_one_3bit_1.qasm"
cir = load_circuit(path)
cir_t = circuit_trans(cir, 5)
sim = simu_cir(cir_t)
print(sim)
print(cir_t.qasm())
