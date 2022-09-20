import os
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import CX, H
from QuICT.core.layout.layout import Layout
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

from qiskit import QuantumCircuit, transpile
# from qiskit.transpiler import PassManager
# from qiskit.transpiler import CouplingMap
# from qiskit.transpiler.passes import  StochasticSwap
# from qiskit.qasm import Qasm
# from QuICT.qcda.mapping.mcts.mcts_mapping import MCTSMapping

# coupling = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]


# layout = Layout.load_file(
#     os.path.dirname(os.path.abspath(__file__)) + 
#     f"/../layout/line7.layout"
# )


# cir_q = Qasm("wr_unit_test/qcda-benchmark/mapping/random_circuit_for_correction.qasm").parse()
# cir_q = QuantumCircuit.from_qasm_file("wr_unit_test/qcda-benchmark/mapping/random_circuit_for_correction.qasm")
# print(cir_q)
cir_q = QuantumCircuit.from_qasm_file("wr_unit_test/qcda-benchmark/mapping/random_circuit_for_correction.qasm")
print(cir_q)
# coupling_map = CouplingMap(couplinglist=coupling)
# ss = StochasticSwap(coupling_map=coupling_map)
# pass_manager = PassManager(ss)
# stochastic_circ = pass_manager.run(cir_q)

circ = transpile(circuits=cir_q, optimization_level=1)
print(circ.size())

# print(stochastic_circ.draw())