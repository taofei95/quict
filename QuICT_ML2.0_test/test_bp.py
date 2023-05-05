import torch
import time

from QuICT.algorithm.quantum_machine_learning.utils_v1 import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils_v1 import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils_v1.gate_tensor import *
from QuICT.algorithm.quantum_machine_learning.utils_v1 import GpuSimulator
from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator
from QuICT.core.gate.utils import Variable
from QuICT.core.circuit import Circuit
from QuICT.core.gate import *
from QuICT.simulation.utils import GateSimulator
from QuICT.simulation.state_vector import StateVectorSimulator


def loss_func(state, n_qubits, device=torch.device("cuda:0")):
    if isinstance(state, np.ndarray):
        state = torch.from_numpy(state).to(device)
    hamiltonian = Hamiltonian([[1, "Y1"]])

    ansatz_list = hamiltonian.construct_hamiton_ansatz(n_qubits, device)
    coefficients = hamiltonian.coefficients
    state_vector = torch.zeros(1 << n_qubits, dtype=torch.complex128).to(device)
    for coeff, ansatz in zip(coefficients, ansatz_list):
        sv = ansatz.forward(state)
        state_vector += coeff * sv
    loss = torch.sum(state.conj() * state_vector).real

    loss_state = torch.autograd.grad(loss, state, retain_graph=True)

    return loss


def test_fp_bp(n_qubit, pargs):
    init_state = np.array(
        [
            0.0730 + 1.6100e-01j,
            0.1641 + 6.5849e-02j,
            0.1768 + 3.9029e-18j,
            0.1768 - 1.5306e-17j,
            0.1768 - 1.7781e-17j,
            0.1768 + 2.6655e-18j,
            0.1641 - 6.5849e-02j,
            0.1641 + 6.5849e-02j,
            0.1641 + 6.5849e-02j,
            0.1641 - 6.5849e-02j,
            0.1277 - 1.2222e-01j,
            0.1277 - 1.2222e-01j,
            0.1768 - 1.6543e-17j,
            0.1768 - 2.2615e-17j,
            0.1641 - 6.5849e-02j,
            0.1641 + 6.5849e-02j,
            0.1641 + 6.5849e-02j,
            0.1641 - 6.5849e-02j,
            0.1768 - 2.2615e-17j,
            0.1768 - 1.6543e-17j,
            0.1277 - 1.2222e-01j,
            0.1277 - 1.2222e-01j,
            0.1641 - 6.5849e-02j,
            0.1641 + 6.5849e-02j,
            0.1641 + 6.5849e-02j,
            0.1641 - 6.5849e-02j,
            0.1768 + 2.6655e-18j,
            0.1768 - 1.7781e-17j,
            0.1768 - 1.5306e-17j,
            0.1768 + 3.9029e-18j,
            0.1641 + 6.5849e-02j,
            0.0730 + 1.6100e-01j,
        ]
    )
    simulator = GpuSimulator()
    params = torch.nn.Parameter(
        torch.tensor(pargs, device=torch.device("cuda:0")), requires_grad=True
    )
    ansatz = Ansatz(n_qubit)
    ansatz.add_gate(H_tensor)
    ansatz.add_gate(Rx_tensor(params[0]), 2)
    ansatz.add_gate(Rx_tensor(2 * params[1]), 1)
    ansatz.add_gate(Rx_tensor(params[2]), 0)
    ansatz.add_gate(Rx_tensor(params[3] / 3 - 1), 3)
    ansatz.add_gate(Rx_tensor(params[4] ** 2), 1)
    ansatz.add_gate(Rzx_tensor(params[1]), [2, 1])

    variables = Variable(np.array(pargs))
    circuit = Circuit(n_qubit)
    H | circuit
    Rx(variables[0]) | circuit(2)
    Rx(2 * variables[1]) | circuit(1)
    Rx(variables[2]) | circuit(0)
    Rx(variables[3] / 3 - 1) | circuit(3)
    Rx(variables[4] ** 2) | circuit(1)
    Rzx(variables[1]) | circuit([2, 1])

    print("--------------GPUSimulator-----------------")
    state = simulator.forward(ansatz, state=init_state.copy())
    loss = loss_func(state, n_qubit)
    loss.backward()
    print(params.grad)

    print("--------------Adjoint-----------------")

    simulator = StateVectorSimulator(device="GPU")
    sv = simulator.run(circuit, init_state.copy())

    differ = Differentiator()
    h = Hamiltonian([[1, "Y1"]])
    variables, _ = differ.run(circuit, variables, sv, h)
    print(variables.grads)


if __name__ == "__main__":
    test_fp_bp(5, [1.8, 2.2, -0.43, 9.9, 1])

