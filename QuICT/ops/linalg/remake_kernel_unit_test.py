import numpy as np
from numba import cuda as nb_cuda

from time import time

from QuICT.core import *
from QuICT.ops.linalg.cuda_kernel_remake import state_vector_simulation
from QuICT.algorithm.amplitude import Amplitude


def test_kernel():
    qubit_num = 30
    initial_state = np.zeros(shape=1 << qubit_num, dtype=np.complex64)
    initial_state[0] = 1.0 + 0.0j
    circuit = Circuit(qubit_num)
    # circuit.random_append(500)
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    # QFT.build_gate(qubit_num) | circuit
    # QFT.build_gate(qubit_num) | circuit
    # QFT.build_gate(qubit_num) | circuit

    with nb_cuda.defer_cleanup():
        start_time = time()
        # state = state_vector_simulation(initial_state, circuit.gates)
        state_vector_simulation(initial_state, circuit.gates)
        end_time = time()
        cuda_duration = end_time - start_time

    start_time = time()
    # state_expected = Amplitude.run(circuit)
    end_time = time()
    old_algo_duration = end_time - start_time

    # assert np.allclose(state, state_expected)

    print()
    print(f"Old algo: {old_algo_duration:0.4f} s")
    print(f"Remade cuda: {cuda_duration:0.4f} s")
