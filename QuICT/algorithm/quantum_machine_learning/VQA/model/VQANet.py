import torch
import numpy as np
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


class VQANet(torch.nn.Module):
    def __init__(
        self,
        n_qubits: int,
        depth: int,
        hamiltonian: Hamiltonian,
        device=torch.device("cuda:0"),
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.hamiltonian = hamiltonian
        self.device = device
        self.define_network()

    def define_network(self):
        raise NotImplementedError

    def loss_func(self, state, simulator=ConstantStateVectorSimulator()):
        assert 1 << self.n_qubits == len(state)
        circuits = self.hamiltonian.construct_hamiton_circuit(self.n_qubits)
        coefficients = self.hamiltonian.coefficients
        state_vector = np.zeros(len(state))
        for coeff, circuit in zip(coefficients, circuits):
            sv = simulator.run(circuit, state)
            state_vector += coeff * sv.get().real
        loss = sum(state.get() * state_vector)
        loss = torch.tensor([float(loss)], requires_grad=True)
        return loss
