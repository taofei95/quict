import numpy as np
from numba import cuda as nb_cuda

from time import time

from QuICT.core import *
from QuICT.ops.linalg.cuda_kernel_remake import state_vector_simulation_remake
from QuICT.algorithm.amplitude import Amplitude

import pytest


def test_perf_kernel():
    qubit_num = 30
    initial_state = np.zeros(shape=1 << qubit_num, dtype=np.complex64)
    initial_state[0] = 1.0 + 0.0j

    print()
    print("JIT for the first time. Ignore all output of this section.")
    trash = Circuit(4)
    CRz(1) | trash([0, 1])
    H | trash(0)
    # CCX | trash([0, 1, 2])
    state_vector_simulation_remake(initial_state, trash.gates, True)

    print("Real results starts here")

    circuit = Circuit(qubit_num)

    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    QFT.build_gate(qubit_num) | circuit
    # QFT.build_gate(qubit_num) | circuit
    # QFT.build_gate(qubit_num) | circuit
    # QFT.build_gate(qubit_num) | circuit

    with nb_cuda.defer_cleanup():
        start_time = time()
        # state = state_vector_simulation(initial_state, circuit.gates)
        state_vector_simulation_remake(initial_state, circuit.gates)
        end_time = time()
        cuda_duration = end_time - start_time

    print(f"Remade cuda: {cuda_duration:0.4f} s")


@pytest.mark.parametrize("ignore", range(20))
def test_kernel_correctness(ignore):
    qubit_num = 10
    circuit = Circuit(qubit_num)

    # Rx(1.756) | circuit(0)
    # Rx(3.043) | circuit(1)
    # Rx(1.816) | circuit(1)
    # Rx(1.388) | circuit(0)

    circuit.random_append(rand_size=500, typeList=[
        GATE_ID["X"],
        GATE_ID["Y"],
        GATE_ID["Z"],
        GATE_ID["H"],
        GATE_ID["Rx"],
        GATE_ID["Ry"],
        GATE_ID["Rz"],
        GATE_ID["CX"],
        GATE_ID["CRz"],
    ])

    initial_state = np.zeros(shape=1 << qubit_num, dtype=np.complex64)
    initial_state[0] = 1.0 + 0.0j

    state_expected = Amplitude.run(circuit)

    state = state_vector_simulation_remake(initial_state, circuit.gates)

    assert np.allclose(state, state_expected, atol=1e-5)
