from QuICT.core import Circuit
from QuICT.core.gate import CLIFFORD_GATE_SET
from QuICT.qcda.synthesis.clifford import CliffordUnidirectionalSynthesizer
from QuICT.qcda.optimization.symbolic_clifford_optimization import SymbolicCliffordOptimization


if __name__ == '__main__':
    n = 3
    circuit = Circuit(n)
    circuit.random_append(10 * n, CLIFFORD_GATE_SET)
    CUS = CliffordUnidirectionalSynthesizer()
    SCO = SymbolicCliffordOptimization()
    circuit_opt = CUS.execute(circuit)
    circuit_opt_opt = SCO.execute(circuit_opt)
    print("The original Quantum Circuit.")
    circuit.draw(method="command")
    print("The Quantum Circuit through the CliffordUnidirectionalSynthesizer.")
    circuit_opt.draw(method="command", flatten=True)
    print("The Quantum Circuit through the SymbolicCliffordOptimization")
    circuit_opt_opt.draw(method="command", flatten=True)
