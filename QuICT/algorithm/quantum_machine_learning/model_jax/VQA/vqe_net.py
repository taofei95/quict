import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random

from QuICT.algorithm.quantum_machine_learning.utils import GpuSimulator
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.tools.exception.algorithm import *


class VQENet:
    """Variational Quantum Eigensolver algorithm.

    VQE <https://arxiv.org/abs/1304.3061> is a quantum algorithm that uses a variational
    technique to find the minimum eigenvalue of the Hamiltonian of a given system.
    """

    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        key,
        dtype=jnp.complex64,
        device="gpu:0",
    ):
        """Initialize a VQENet instance.

        Args:
            n_qubits (int): The number of qubits.
            p (int): The number of layers of the network.
            hamiltonian (Hamiltonian): The hamiltonian for a specific task.
            device (torch.device, optional): The device to which the VQANet is assigned.
                Defaults to torch.device("cuda:0").
        """
        self.n_qubits = n_qubits
        self.p = p
        self.hamiltonian = hamiltonian
        self.key = key
        device_type = jax.devices("gpu") if device[:3] == "gpu" else jax.devices("cpu")
        self.device = device_type[int(device[4:])]
        # self.simulator = GpuSimulator()
        self.cir_simulator = ConstantStateVectorSimulator()
        self.dtype = dtype
        self.define_network()

    def define_network(self):
        """Define the network construction.

        Raises:
            NotImplementedError: to be completed
        """
        raise NotImplementedError

    def loss_func(self, state):
        """The loss function for VQE, which aims to minimize the expectation of H.

        Args:
            state (torch.Tensor): The state vector.

        Returns:
            torch.Tensor: Loss, which is equal to the expectation of H.
        """
        if isinstance(state, np.ndarray):
            state = jnp.asarray(state)
            state = jax.device_put(state, self.device)
        if state.shape[0] != 1 << self.n_qubits:
            raise VQEModelError(
                "The input state vector must match the number of qubits."
            )

        circuit_list = self.hamiltonian.construct_hamiton_circuit(self.n_qubits)
        coefficients = self.hamiltonian.coefficients
        state_vector = jnp.zeros(1 << self.n_qubits, dtype=self.dtype)
        state_vector = jax.device_put(state_vector, self.device)

        for coeff, circuit in zip(coefficients, circuit_list):
            sv = self.cir_simulator.run(circuit, state)
            state_vector += coeff * sv
        loss = jnp.sum(state.conj() * state_vector).real

        return loss
