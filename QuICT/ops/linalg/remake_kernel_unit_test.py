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

    print()
    print("JIT for the first time. Ignore all output of this section.")
    trash = Circuit(4)
    CX | trash([0, 1])
    X | trash(0)
    CCX | trash([0, 1, 2])
    state_vector_simulation(initial_state, trash.gates)

    print("Real results starts here")

    circuit = Circuit(qubit_num)

    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit

    with nb_cuda.defer_cleanup():
        start_time = time()
        # state = state_vector_simulation(initial_state, circuit.gates)
        state_vector_simulation(initial_state, circuit.gates)
        end_time = time()
        cuda_duration = end_time - start_time

    print(f"Remade cuda: {cuda_duration:0.4f} s")
