import os
from copy import deepcopy
from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.impl_wrapper import MCTSMapping
from QuICT.qcda.mapping.mcts_legacy import MCTSMapping as MCTSMappingLegacy
from QuICT.qcda.mapping.mcts_refactor import MCTSMapping as MCTSMappingRefactor
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
# qcda_workflow.add_default_mapping(layout)

single_typelist = [GateType.h, GateType.cx, GateType.rz]
pro_typelist = [0.2,0.2,0.6]
qubit_num = 5
cir = Circuit(qubit_num)
cir.random_append(rand_size=20, typelist=single_typelist, probabilities=pro_typelist, random_params=True)

# sim = UnitarySimulator()
# amp1 = sim.run(cir)
# print(abs(amp1))

MCTSMapping = MCTSMappingRefactor
cir_ori = MCTSMapping(layout).execute(deepcopy(cir))
circuit_ori = Circuit(5)
cir_ori | circuit_ori

sim = UnitarySimulator()
amp1 = sim.run(circuit_ori)
print(abs(amp1))

cir_opt = qcda_workflow.compile(deepcopy(cir_ori))
cir_map = MCTSMapping(layout).execute(cir_opt)
circuit_opt = Circuit(5)
cir_map | circuit_opt




print(circuit_ori.qasm())
print(circuit_opt.qasm())

print(circuit_ori.size(), circuit_opt.size())
print(circuit_ori.depth(), circuit_opt.depth())