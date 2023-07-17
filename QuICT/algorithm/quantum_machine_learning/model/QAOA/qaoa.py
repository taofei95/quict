from numpy_ml.neural_nets.optimizers import *

from ..model import Model

from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *


class QAOA(Model):
    """The quantum approximate optimization algorithm (QAOA)."""

    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        optimizer: OptimizerBase,
        params: np.ndarray = None,
        device: str = "GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        """Initialize a QAOA model.

        Args:
            n_qubits (int): The number of qubits.
            p (int): The number of layers of the QAOA network.
            hamiltonian (Hamiltonian): The hamiltonian for a specific problem.
            optimizer (OptimizerBase): The optimizer used to optimize the network.
            params (np.ndarray, optional): Initialization parameters. Defaults to None.
            device (str, optional): The device type, one of [CPU, GPU]. Defaults to "GPU".
            gpu_device_id (int, optional): The GPU device ID. Defaults to 0.
            differentiator (str, optional): The differentiator type, one of ["adjoint", "parameter_shift]. Defaults to "adjoint".
        """
        super(QAOA, self).__init__(
            n_qubits,
            optimizer,
            hamiltonian,
            params,
            device,
            gpu_device_id,
            differentiator,
        )
        self._qaoa_builder = QAOALayer(n_qubits, p, hamiltonian)
        self._circuit = self._qaoa_builder.init_circuit(params=params)
        self._params = self._qaoa_builder.params

    def run_step(self):
        """Train QAOA for one step.

        Returns:
            np.ndarry: The state vector.
            np.float: The loss for this step.
        """
        # FP
        state = self._simulator.run(self._circuit)
        # BP
        _, loss = self._differentiator.run(
            self._circuit, self._params, state, -1 * self._hamiltonian
        )
        # optimize
        self._params.pargs = self._optimizer.update(
            self._params.pargs, self._params.grads, "QAOA_params"
        )
        self._params.zero_grad()
        # update
        self._update()
        return state, loss

    def sample(self, shots: int):
        """Sample the measured result from current state vector.

        Args:
            shots (int): The sample times for current state vector.

        Returns:
             List[int]: The measured result list.
        """
        sample = self._simulator.sample(shots)
        return sample

    def _update(self):
        self._circuit = self._qaoa_builder.init_circuit(self._params)
