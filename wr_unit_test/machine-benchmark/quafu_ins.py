import os
from copy import deepcopy
from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.mcts.mcts_mapping import MCTSMapping
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.qcda.synthesis import InstructionSet
from QuICT.simulation.unitary import *
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout
from QuICT.qcda.qcda import QCDA

qf_iset = InstructionSet(
    GateType.cx,
    [GateType.h, GateType.rx, GateType.ry, GateType.rz]
)
layout = Layout.load_file(
    os.path.dirname(os.path.abspath(__file__)) + 
    f"/line5.layout"
)
qcda_workflow = QCDA()
# qcda_workflow.add_default_synthesis(qf_iset)
qcda_workflow.add_default_optimization()
qcda_workflow.add_default_mapping(layout)

single_typelist = [GateType.h, GateType.cx, GateType.rz]
pro_typelist = [0.2,0.2,0.6]
qubit_num = 5
cir = Circuit(qubit_num)
cir.random_append(rand_size=50, typelist=single_typelist, probabilities=pro_typelist)

cir_ori = MCTSMapping(layout).execute(deepcopy(cir))
cir_opt = qcda_workflow.compile(deepcopy(cir))
# cir_map = MCTSMapping(layout).execute(cir_opt)


sim = UnitarySimulator()
amp1 = sim.run(cir_ori)
print(abs(amp1))
print("\n")
amp = sim.run(cir_opt)
print(abs(amp))

print(cir_ori.qasm())
print(cir_opt.qasm())

print(cir_ori.size(), cir_opt.size())
print(cir_ori.depth(), cir_opt.depth())