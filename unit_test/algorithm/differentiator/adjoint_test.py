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


def loss_func(state, n_qubits, hamiltonian, device=torch.device("cuda:0")):
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).to(device)
    ansatz_list = hamiltonian.construct_hamiton_ansatz(n_qubits, device)
    coefficients = hamiltonian.coefficients
    state_vector = torch.zeros(1 << n_qubits, dtype=torch.complex128).to(device)
    for coeff, ansatz in zip(coefficients, ansatz_list):
        sv = ansatz.forward(state)
        state_vector += coeff * sv
    loss = torch.sum(state.conj() * state_vector).real
    return loss


def test_fp_bp(n_qubit, pargs):
    simulator = GpuSimulator()
    h = Hamiltonian([[1, "Y1"]])
    params = torch.nn.Parameter(
        torch.tensor(pargs, device=torch.device("cuda:0")), requires_grad=True
    )
    ansatz = Ansatz(n_qubit)
    ansatz.add_gate(H_tensor)
    ansatz.add_gate(Rx_tensor(0.2), 1)
    ansatz.add_gate(Rzx_tensor(-0.2 * params[1] ** 3 + 0.6), [0, 1])
    ansatz.add_gate(Rzz_tensor(params[1]), [0, 1])
    ansatz.add_gate(Rzx_tensor(params[2]), [0, 1])
    ansatz.add_gate(Rzx_tensor(3), [0, 1])

    variables = Variable(np.array(pargs))
    circuit = Circuit(n_qubit)
    H | circuit
    Rx(0.2) | circuit(1)
    Rzx(-0.2 * variables[1] ** 3 + 0.6) | circuit([0, 1])
    Rzz(variables[1]) | circuit([0, 1])
    Rzx(variables[2]) | circuit([0, 1])
    Rzx(3) | circuit([0, 1])

    print("--------------GPUSimulator-----------------")

    start = time.time()
    state = simulator.forward(ansatz, state=None)
    loss = loss_func(state, n_qubit, h)
    loss.backward()
    # print("FP + BP", time.time() - start)
    print(params)
    print(params.grad)

    print("--------------Adjoint-----------------")

    simulator = StateVectorSimulator(device="GPU")
    differ = Adjoint(device="GPU")

    start = time.time()
    sv = simulator.run(circuit)
    differ.run(circuit, variables, sv, h)
    # print("FP + BP", time.time() - start)
    print(variables.pargs)
    print(variables.grads)
    
    variables.pargs = np.array([1, 1, 1])
    print(variables.pargs)
    circuit.update(variables)
    
    for gate in circuit.gates:
        if isinstance(gate.parg, Variable):
            print(gate.parg.pargs)

    return variables


if __name__ == "__main__":
    variables = test_fp_bp(3, [1.8, -0.7, 2.3])
    
