from ..model import Model

from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *


class QAOA(Model):
    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian = None,
        params: np.ndarray = None,
        device: str = "GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        super(QAOA, self).__init__(
            n_qubits, hamiltonian, params, device, gpu_device_id, differentiator
        )
        assert hamiltonian is not None
        self._qaoa_builder = QAOALayer(n_qubits, p, hamiltonian)
        self._circuit = self._qaoa_builder.init_circuit(params=params)
        self._params = self._qaoa_builder.params

    def run_step(self, optimizer):
        # FP
        state = self._simulator.run(self._circuit)
        # BP
        _, loss = self._differentiator.run(
            self._circuit, self._params, state, -1 * self._hamiltonian
        )
        # optimize
        self._params.pargs = optimizer.update(
            self._params.pargs, self._params.grads, "QAOA_params"
        )
        self._params.zero_grad()
        # update
        self.update()
        return state, loss

    def update(self):
        self._circuit = self._qaoa_builder.init_circuit(self._params)

    def sample(self, shots):
        sample = self._simulator.sample(shots)
        return sample
