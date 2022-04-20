import numpy as np
from scipy.stats import unitary_group
from QuICT.core import Circuit

from QuICT.quantum_state_preparation import QuantumStatePreparation
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


def test_with_uniformly_gates():
    for n in range(2, 6):
        for _ in range(100):
            state_vector = unitary_group.rvs(1 << n)[0]
            gates = QuantumStatePreparation.with_uniformly_gates(state_vector)
            circuit = Circuit(n)
            circuit.extend(gates)
            simulator = ConstantStateVectorSimulator()
            state = simulator.run(circuit)
            assert np.allclose(state_vector, state)
