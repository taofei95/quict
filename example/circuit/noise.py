import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import GateType, H
from QuICT.core.noise import (
    BitflipError, DampingError, DepolarizingError, PauliError, PhaseflipError,
    NoiseModel, ReadoutError
)
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulation
from QuICT.core.noise.utils import is_kraus_ops


def build_circuit_with_dampling_noise(circuit):
    amp_err = DampingError(amplitude_prob=0.1, phase_prob=0, dissipation_state=0.4)
    print(amp_err.operators)
    print(len(amp_err.operators))
    phase_err = DampingError(amplitude_prob=0, phase_prob=0.3)
    print(phase_err.operators)
    amp_phase_err = DampingError(amplitude_prob=0.1, phase_prob=0.3, dissipation_state=0.5)

    # build noise model
    nm = NoiseModel()
    nm.add_noise_for_all_qubits(amp_err, ['h', 'u1', 'rx'])
    nm.add_noise_for_all_qubits(phase_err, ['t', 's'])
    nm.add_noise_for_all_qubits(amp_phase_err, ['x'])

    # dm_simu = DensityMatrixSimulation(accumulated_mode=False)
    # quict_result = dm_simu.run(circuit, noise_model=nm)
    # print(quict_result)


def build_circuit_with_pauil_noise(circuit):
    pass


def build_circuit_with_depolarizing_noise(circuit):
    depolarizing_rate = 0.05
    # 1-qubit depolarizing error
    single_dep = DepolarizingError(depolarizing_rate, num_qubits=1)
    # 2-qubit depolarizing error
    double_dep = DepolarizingError(depolarizing_rate, num_qubits=2)

    nm = NoiseModel()
    nm.add_noise_for_all_qubits(single_dep, ['h', 'u1', 'rx'])
    nm.add(double_dep, ['cx'], [0, 1])

    dm_simu = DensityMatrixSimulation()
    quict_result = dm_simu.run(circuit, noise_model=nm)


def build_circuit_with_readout_noise(circuit):
        ###########################test_readout##########################################
    # single-qubit Readout Error
    single_readout = ReadoutError(np.array([[0.8, 0.2], [0.2, 0.8]]))

    nm = NoiseModel()
    nm.add_readout_error(single_readout, 4)

    dm_simu = DensityMatrixSimulation()
    quict_result = dm_simu.run(circuit, noise_model=nm)


if __name__ == "__main__":
    typelist = [GateType.h, GateType.x, GateType.s, GateType.y, GateType.t, GateType.cx, GateType.crz, GateType.cu1]
    cir = Circuit(4)
    H | cir
    cir.random_append(typelist = typelist)

    build_circuit_with_dampling_noise(cir)
