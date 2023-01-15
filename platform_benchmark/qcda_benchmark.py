import os
from QuICT.core.circuit.circuit import Circuit

from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import CLIFFORD_GATE_SET, GateType
from QuICT.qcda.mapping import *
from QuICT.qcda.mapping.ai.rl_mapping import RlMapping
from QuICT.qcda.optimization import *
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.clifford.clifford_synthesizer import CliffordUnidirectionalSynthesizer
from QuICT.qcda.synthesis.gate_transform.gate_transform import *

########mapping##########################################################
layout_path = os.path.join(os.path.dirname(__file__), "../layout/ibmqx2_layout.json")
layout = Layout.load_file(layout_path)
circuit = Circuit(5)
circuit.random_append(50, typelist=[GateType.cx])

# mcts
mcts = MCTSMapping(layout)
circuit_map = mcts.execute(circuit)
# sabre
sabre = SABREMapping(layout)
circuit_map = sabre.execute(circuit)
# rl
mapper = RlMapping(layout=layout)
mapped_circ = mapper.execute(circuit)

#########synthesis###############################################################3
# unitarydecomposition

# gate transform
qcda = QCDA()
InSet = [GoogleSet, IBMQSet, IonQSet, NamSet, OriginSet, USTCSet]
for inset in InSet:
    qcda.add_gate_transform(inset)

# Clifford synthesizer
circuit = Circuit(5)
circuit.random_append(20 * 5, CLIFFORD_GATE_SET)
CUS = CliffordUnidirectionalSynthesizer()
circuit_opt = CUS.execute(circuit)

# quantum state preparation 


#########optimization########################################################
# symbolic clifford opt
circuit = Circuit(5)
circuit.random_append(20 * 5, CLIFFORD_GATE_SET)
CUS = CliffordUnidirectionalSynthesizer()
SCO = SymbolicCliffordOptimization()
circuit_opt = CUS.execute(circuit)
circuit_opt_opt = SCO.execute(circuit_opt)

# cnot ancilla
CA = CnotAncilla(size=1)
result_circuit = CA.execute(circuit)
    
# commutative opt
typelist = [
    GateType.rx, GateType.ry, GateType.rz, GateType.x,
    GateType.y, GateType.z, GateType.cx
]
circuit = Circuit(5)
circuit.random_append(rand_size=100, typelist=typelist, random_params=True)
circuit.draw()

CO = CommutativeOptimization()
circuit_opt = CO.execute(circuit)

# template opt
circuit = Circuit(5)
typelist = [GateType.x, GateType.cx, GateType.ccx,
            GateType.h, GateType.s, GateType.t, GateType.sdg, GateType.tdg]
circuit.random_append(200, typelist=typelist)
TO = TemplateOptimization()
circ_optim = TO.execute(circuit)

