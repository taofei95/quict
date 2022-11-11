from quafu import QuantumCircuit, User
from quafu import Task

import os
from copy import deepcopy

from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.mcts_refactor import MCTSMapping as MCTSMappingRefactor
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.qcda.qcda import QCDA
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout
from QuICT.qcda.synthesis import InstructionSet
from QuICT.qcda.synthesis.gate_transform.special_set.nam_set import NamSet



user = User()
user.save_apitoken('Uro0u29JJUKmTPHKwredqlXLP7a1VR22i5zQPo0z8Wt.9VTOyQzM1kjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

layout = Layout.load_file(
    os.path.dirname(os.path.abspath(__file__)) + 
    f"/layout/line6.layout"
)

def load_circuit(path):
    from QuICT.tools.interface.qasm_interface import OPENQASMInterface
    cir = OPENQASMInterface.load_file(path).circuit

    return cir

def circuit_trans(cir):
    qcda_workflow = QCDA()
    qcda_workflow.add_default_synthesis(NamSet)
    qcda_workflow.add_default_optimization()
    # qcda_workflow.add_default_mapping(layout)
    cg_inst = qcda_workflow.compile(cir)
    return cg_inst

# def circuit_opt(cir):
#     # mapping
#     MCTSMapping = MCTSMappingRefactor
#     cir_map = MCTSMapping(layout).execute(deepcopy(cir))
#     circuit_map = Circuit(3)
#     cir_map | circuit_map
#     circuit_map.gate_decomposition()
#     #opt
#     cir_opt = AutoOptimization().execute(circuit_map)
#     return cir_opt

def quafu_run(cir, name:str):
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

def ins_qcda(circuit):
    qcda = QCDA()
    qcda.add_default_synthesis(NamSet)
    circuit_phy = qcda.compile(circuit)
    return circuit_phy
    
    
path = "wr_unit_test/machine-benchmark/qasm/adder_multiplier/plus_one_4bit_1.qasm"
cir = load_circuit(path)
cir_t = opt(cir)
# print(quafu_run(cir_t,"add_2bit_2"))

# amp = simu_cir(cir_t)
# print(f'add_2bit_2')
# print(abs(amp))
print(cir_t.qasm())

print(cir.size(), cir.depth())
