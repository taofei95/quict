from QuICT.core import Circuit
from QuICT.core.gate import CLIFFORD_GATE_SET
from QuICT.qcda.synthesis.clifford import CliffordUnidirectionalSynthesizer
from QuICT.qcda.optimization.symbolic_clifford_optimization import SymbolicCliffordOptimization

if __name__ == '__main__':
    n = 5
    circuit = Circuit(n)
    circuit.random_append(20 * n, CLIFFORD_GATE_SET)
    CUS = CliffordUnidirectionalSynthesizer()
    SCO = SymbolicCliffordOptimization()
    circuit_opt = CUS.execute(circuit)
    circuit_opt_opt = SCO.execute(circuit_opt)
    circuit.draw()
    circuit_opt.draw()
    circuit_opt_opt.draw()
