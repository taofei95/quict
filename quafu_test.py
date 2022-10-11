#build env
from quafu import QuantumCircuit
from quafu import User
user = User()
user.save_apitoken('AnNvhF_3fbwech2me9FWr_gZLKJor01-qVibchCtqHP.9hjM4YjNwgjN2EjOiAHelJCL3QTM6ICZpJye.9JiN1IzUIJiOicGbhJCLiQ1VKJiOiAXe0Jye')

#build qasm
import os
from copy import deepcopy
from QuICT.core.circuit.circuit import Circuit
from QuICT.qcda.mapping.impl_wrapper import MCTSMapping
from QuICT.qcda.mapping.mcts_refactor import MCTSMapping as MCTSMappingRefactor
from QuICT.qcda.optimization.auto_optimization import AutoOptimization
from QuICT.qcda.synthesis.gate_decomposition.gate_decomposition import GateDecomposition
from QuICT.simulation.unitary import *
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout

layout = Layout.load_file(
    os.path.dirname(os.path.abspath(__file__)) + 
    f"/wr_unit_test/qcda-benchmark/layout/line5.layout"
)

single_typelist = [GateType.h, GateType.cx, GateType.rz]
pro_typelist = [0.2,0.2,0.6]
qubit_num = 5
cir = Circuit(qubit_num)
cir.random_append(rand_size=50, typelist=single_typelist, probabilities=pro_typelist, random_params=True)

#mapping
MCTSMapping = MCTSMappingRefactor
cir_map = MCTSMapping(layout).execute(deepcopy(cir))
circuit_map = Circuit(5)
cir_map | circuit_map
#transform
cir_trans = GateDecomposition().execute(circuit_map)
#opt
cir_opt = AutoOptimization().execute(cir_trans)

# sim = UnitarySimulator()
# amp1 = sim.run(cir)
# print(abs(amp1))
# print("\n")
# amp1 = sim.run(cir_opt)
# print(abs(amp1))

# print(cir.qasm())
# print(cir_opt.qasm())

# print(cir.size(), cir_opt.size())
# print(cir.depth(), cir_opt.depth())

#submit
# b = open('test/test.qasm', 'rb').read()
cir_qasm = str(b'cir.qasm()', encoding = "utf-8")
cir_opt_qasm = str(b'cir_opt.qasm()', encoding = "utf-8")

qc = QuantumCircuit(5)
test_cir = cir_qasm
qc.from_openqasm(test_cir)

qcc = QuantumCircuit(5)
test_ciropt = cir_opt_qasm
qcc.from_openqasm(test_ciropt)
# qc.draw_circuit()

from quafu import Task
task = Task()
task.load_account()
task.config(backend="ScQ-P10", shots=3000, compile=True, priority=2)
# res = task.send(qc, wait=False, name=11, group="Q3_T1")
res = task.send(qc, name=1)
print(res.counts) #counts
print(res.amplitudes) #amplitude
res.plot_amplitudes()

task = Task()
task.load_account()
task.config(backend="ScQ-P10", shots=3000, compile=True, priority=2)
# res = task.send(qc, wait=False, name=11, group="Q3_T1")
res = task.send(qc, name=2)
print(res.counts) #counts
print(res.amplitudes) #amplitude
res.plot_amplitudes()