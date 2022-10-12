import os
from copy import deepcopy
from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.impl_wrapper import MCTSMapping
from QuICT.qcda.mapping.mcts_legacy import MCTSMapping as MCTSMappingLegacy
from QuICT.qcda.mapping.mcts_refactor import MCTSMapping as MCTSMappingRefactor
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.qcda.synthesis import InstructionSet
from QuICT.qcda.synthesis.gate_decomposition.gate_decomposition import GateDecomposition
from QuICT.simulation.unitary import *
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout
from QuICT.qcda.qcda import QCDA

# qf_iset = InstructionSet(
#     GateType.cx,
#     [GateType.h, GateType.rx, GateType.ry, GateType.rz]
# )
layout = Layout.load_file(
    os.path.dirname(os.path.abspath(__file__)) + 
    f"/line5.layout"
)
# qcda_workflow = QCDA()
# qcda_workflow.add_default_synthesis(qf_iset)
# qcda_workflow.add_default_optimization()
# qcda_workflow.add_default_mapping(layout)

single_typelist = [GateType.h, GateType.cx, GateType.rz]
pro_typelist = [0.2,0.2,0.6]
qubit_num = 5
cir = Circuit(qubit_num)
cir.random_append(rand_size=50, typelist=single_typelist, probabilities=pro_typelist, random_params=True)

# sim = UnitarySimulator()
# amp1 = sim.run(cir)
# print(abs(amp1))

# mapping
MCTSMapping = MCTSMappingRefactor
cir_map = MCTSMapping(layout).execute(deepcopy(cir))
circuit_map = Circuit(5)
cir_map | circuit_map
#transform
cir_trans = GateDecomposition().execute(circuit_map)
#opt
cir_opt = AutoOptimization().execute(cir_trans)



print(cir.qasm())
print(cir_opt.qasm())

print(cir.size(), cir_opt.size())
print(cir.depth(), cir_opt.depth())