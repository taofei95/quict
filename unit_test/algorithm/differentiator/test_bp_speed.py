import torch
import time

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *
from QuICT.algorithm.quantum_machine_learning.utils import GpuSimulator
from QuICT.algorithm.quantum_machine_learning.differentiator import Adjoint
from QuICT.core.gate.utils import Variable
from QuICT.core.circuit import Circuit
from QuICT.core.gate import *
from QuICT.simulation.utils import GateSimulator
from QuICT.simulation.state_vector import StateVectorSimulator


def test_fp_bp(n_qubit, layers):
    variables = Variable(np.random.rand(len(layers), n_qubit - 1))

    circuit = Circuit(n_qubit)
    H | circuit
    for l, gate in zip(range(len(layers)), layers):
        for i in range(n_qubit - 1):
            gate(variables[l, i]) | circuit([i, n_qubit - 1])

    print("--------------Adjoint-----------------")

    simulator = StateVectorSimulator(device="GPU")
    start = time.time()
    sv = simulator.run(circuit)
    print("FP", time.time() - start)

    differ = Adjoint(device="GPU")
    h = Hamiltonian([[1, "Y1"]])
    start = time.time()
    differ.run(circuit, variables, sv, h)
    print("BP", time.time() - start)


if __name__ == "__main__":
    # 17qubits FP0.005s, BP0.034s
    # 20qubits FP0.014s, BP0.16s
    # 25qubits FP0.35s, BP4.5s
    
    test_fp_bp(25, [Rzx] * 2)
