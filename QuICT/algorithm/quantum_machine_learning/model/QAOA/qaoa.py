from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.simulation.simulator import Simulator
from QuICT.simulation.state_vector import StateVectorSimulator


class QAOA:
    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        device: str = "GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        self._qaoa_builder = QAOALayer(n_qubits, p, hamiltonian)
        self._circuit = self._qaoa_builder.init_circuit()
        self._params = self._qaoa_builder.params
        self._circuit.gate_decomposition(decomposition=False)
        self._hamiltonian = hamiltonian
        self._simulator = StateVectorSimulator(
            device=device, gpu_device_id=gpu_device_id
        )
        self._differentiator = Differentiator(
            device=device, backend=differentiator, gpu_device_id=gpu_device_id
        )

    def run_step(self, optimizer):
        # FP
        state = self._simulator.run(self._circuit)
        # BP
        _, loss = self._differentiator.run(
            self._circuit, self._params, state, -1 * self._hamiltonian
        )
        # optimize
        optimizer.update(
            param = self._params, param_name = "params"
        )
        self._params.zero_grad()
        # update
        self._circuit = self._qaoa_builder.init_circuit(self._params)
        return state, loss

    def sample(self, shots):
        sample = self._simulator.sample(shots)

        return sample
