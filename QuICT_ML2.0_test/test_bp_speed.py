import time
import numpy_ml

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator
from QuICT.core.circuit import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *


def test_fp_bp(n_qubit, layers):
    variables = Variable(np.random.rand(len(layers), n_qubit - 1))

    circuit = Circuit(n_qubit)
    H | circuit
    for l, gate in zip(range(len(layers)), layers):
        for i in range(n_qubit - 1):
            gate(variables[l, i]) | circuit([i, n_qubit - 1])
    print(circuit.depth())

    print("--------------Adjoint-----------------")

    simulator = StateVectorSimulator(device="GPU")
    start = time.time()
    sv = simulator.run(circuit)
    print("FP", time.time() - start)

    differ = Differentiator(device="GPU")
    optim = numpy_ml.neural_nets.optimizers.Adam(lr=0.1)
    h = Hamiltonian([[1, "Y1"]])
    start = time.time()
    variables, _ = differ.run(circuit, variables, sv, h)
    print("BP", time.time() - start)

    start = time.time()
    variables.pargs = optim.update(variables.pargs, variables.grads, "variables")
    print("OPTIMIZE", time.time() - start)
    variables.zero_grad()
    
    start = time.time()
    circuit.update(variables)
    print("UPDATE", time.time() - start)


if __name__ == "__main__":
    # 17qubits FP0.005s, BP0.034s
    # 20qubits FP0.014s, BP0.16s
    # 25qubits FP0.35s, BP4.5s
    test_fp_bp(17, [Rzx] * 2)
