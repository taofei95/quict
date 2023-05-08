from QuICT.algorithm.quantum_machine_learning.ansatz_library import *
from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator
from QuICT.algorithm.quantum_machine_learning.utils_v1 import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils_v1.ml_utils import *
from QuICT.simulation.simulator import Simulator
from QuICT.simulation.state_vector import StateVectorSimulator


class QAOA:
    def __init__(
        self,
        n_qubits: int,
        p: int,
        hamiltonian: Hamiltonian,
        shots: int = 1000,
        device: str = "GPU",
        gpu_device_id: int = 0,
        simulator: str = "state_vector",
        differentiator: str = "adjoint",
    ):
        qaoa_builder = QAOALayer(n_qubits, p, hamiltonian)
        self._circuit = qaoa_builder.circuit
        self._params = qaoa_builder.params
        self._circuit.gate_decomposition(decomposition=False)
        self._hamiltonian = hamiltonian
        self._shots = shots
        # self._simulator = Simulator(
        #     device=device, backend=simulator, gpu_device_id=gpu_device_id
        # )
        self._simulator = StateVectorSimulator(
            device=device, gpu_device_id=gpu_device_id
        )
        self._differentiator = Differentiator(
            device=device, backend=differentiator, gpu_device_id=gpu_device_id
        )

    def run_step(self, optimizer):
        # result = self._simulator.run(self._circuit, self._shots)
        # state = result["data"]["state_vector"]
        # sample = result["data"]['counts']
        state = self._simulator.run(self._circuit)
        sample = self._simulator.sample(self._shots)
        self._differentiator.run(self._circuit, self._params, state, self._hamiltonian)
        self._params = apply_optimizer(optimizer, self._params)
        self._params.zero_grad()
        self._circuit.update(self._params)
        return state.get(), sample

    def sample(self, state):
        theor_prob = (state * state.conj()).real
        theor_prob = theor_prob / np.sum(theor_prob)
        sample = np.random.choice(
            a=range(len(theor_prob)), size=self._shots, replace=True, p=theor_prob
        )
        idx, counts = np.unique(sample, return_counts=True)
        prob = np.zeros(len(theor_prob))
        for i, count in zip(idx, counts):
            prob[i] = count

        return prob.tolist()
