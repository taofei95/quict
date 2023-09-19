import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import GateType, H, CX
from QuICT.core.noise import (
    BitflipError, DampingError, DepolarizingError, PauliError, PhaseflipError, PhaseBitflipError,
    NoiseModel, ReadoutError
)
from QuICT.simulation.density_matrix import DensityMatrixSimulator


def build_dampling_noise():
    # Amplitude error
    amp_err = DampingError(amplitude_prob=0.2, phase_prob=0, dissipation_state=0.3)

    # Phase Error
    phase_err = DampingError(amplitude_prob=0, phase_prob=0.5)

    # amp-phase error
    amp_phase_err = DampingError(amplitude_prob=0.1, phase_prob=0.5, dissipation_state=0.2)

    # Amp-Phase tensor error
    tensor_err = amp_err.tensor(phase_err)

    # build noise model
    nm = NoiseModel()
    nm.add(tensor_err, ['cx'], [0, 1])

    return nm


def build_pauli_noise():
    pauil_error_rate = 0.4
    # bitflip pauilerror
    bf_err = BitflipError(pauil_error_rate)

    # phaseflip pauilerror
    pf_err = PhaseflipError(pauil_error_rate)

    # bitphaseflip pauilerror
    bpf_err = PhaseBitflipError(pauil_error_rate)

    # 2-bits pauilerror
    bits_err = PauliError(
        [('zy', pauil_error_rate), ('xi', 1 - pauil_error_rate)],
        num_qubits=2
    )

    # bit-phase compose error
    bp_err = bf_err.compose(pf_err)

    # build noise model
    nm = NoiseModel()
    nm.add_noise_for_all_qubits(bp_err, ['h'])
    nm.add_noise_for_all_qubits(bpf_err, ['h'])
    nm.add(bits_err, ['cx'], [0, 1])

    return nm


def build_depolarizing_noise():
    depolarizing_rate = 0.05
    # 1-qubit depolarizing error
    single_dep = DepolarizingError(depolarizing_rate, num_qubits=1)

    # 2-qubit depolarizing error
    double_dep = DepolarizingError(0.01, num_qubits=2)

    nm = NoiseModel()
    nm.add_noise_for_all_qubits(single_dep, ['h', 'u1', 'rx'])
    nm.add(double_dep, ['cx'], [0, 1])

    return nm


def build_readout_noise():
    single_readout = ReadoutError(np.array([[0.8, 0.2], [0.2, 0.8]]))
    double_readout = ReadoutError(
        np.array(
            [[0.7, 0.1, 0.1, 0.1],
             [0.1, 0.7, 0.1, 0.1],
             [0.1, 0.1, 0.7, 0.1],
             [0.1, 0.1, 0.1, 0.7]]
        )
    )

    powed_readout = single_readout.power(3)

    nm = NoiseModel()
    nm.add_readout_error(single_readout, 2)     # add readout error for qubit 4
    nm.add_readout_error(powed_readout, [0, 2, 3])      # add readout error for qubit [0, 2, 3]
    nm.add_readout_error(double_readout, [1, 3])    # add two qubits readout error for qubit [1, 3]

    return nm


if __name__ == "__main__":
    typelist = [GateType.h, GateType.x, GateType.s, GateType.y, GateType.t, GateType.cx, GateType.crz, GateType.cu1]
    cir = Circuit(4)
    H | cir(0)
    for i in range(3):
        CX | cir([i, i + 1])          # append CX gate

    nm = build_readout_noise()

    # 含噪声量子电路
    noised_cir = nm.transpile(cir)

    # 含噪声量子电路模拟
    simulator = DensityMatrixSimulator(accumulated_mode=True)
    sv = simulator.run(cir, quantum_machine_model=nm)
    sample_result = simulator.sample(20)

    print(sample_result)
