import unittest
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.operator import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.qft import QFT, IQFT


class TestOperator(unittest.TestCase):
    def test_checkpoint(self):
        # Build Circuit and CompositeGate
        cir = Circuit(5)
        QFT(5) | cir

        cgate = CompositeGate("x")
        X | cgate(0)
        X | cgate(1)
        cgate1 = CompositeGate("y")
        Y | cgate1(0)
        Y | cgate1(1)

        # Initial CheckPoint and CheckPointChild
        cp = CheckPoint()
        cp_child = cp.get_child(5)

        # add CheckPoint into Circuit
        cp | cir
        IQFT(5) | cir

        # add CompositeGate with cp_child into cir
        cp_child | cgate
        cp_child | cgate1

        cgate | cir
        cgate1 | cir

        # get compositegate name from circuit
        x_gate = cir.gates[-2].name
        y_gate = cir.gates[-1].name

        assert x_gate == "x" and y_gate == "y"

    def test_trigger(self):
        # Build circuit and compositegate
        cir = Circuit(4)
        cgate = CompositeGate()
        H | cgate(0)

        cgate1 = CompositeGate()
        CX | cgate1([0, 1])
        CX | cgate1([1, 2])
        CX | cgate1([2, 3])

        gates = [cgate, cgate1]

        # Build Trigger
        trigger = Trigger(1, gates)
        trigger | cir([0])

        sim = StateVectorSimulator()
        sv = sim.run(cir)

        if cir[0].measured:
            assert np.isclose(sv[-1], 1)

    def test_noise(self):
        from QuICT.core.noise import (
            BitflipError, DampingError, DepolarizingError, PauliError, PhaseflipError, PhaseBitflipError
        )

        cir = Circuit(4)
        error_rate = 0.4
        depolarizing_rate = 0.05

        amp_err = DampingError(amplitude_prob=0.2, phase_prob=0, dissipation_state=0.3)
        bf_err = BitflipError(error_rate)
        pf_err = PhaseflipError(error_rate)
        bpf_err = PhaseBitflipError(error_rate)
        single_dep = DepolarizingError(depolarizing_rate, num_qubits=1)
        bits_err = PauliError(
            [('zy', error_rate), ('xi', 1 - error_rate)],
            num_qubits=1
        )

        gate = gate_builder(GateType.h)

        noisegate = [
            NoiseGate(gate, amp_err), NoiseGate(gate, bf_err),
            NoiseGate(gate, pf_err), NoiseGate(gate, bpf_err),
            NoiseGate(gate, single_dep), NoiseGate(gate, bits_err)
        ]
        for noise in noisegate:
            noise | cir([0])

        assert cir.gates[0].noise_matrix != cir.gates[1].noise_matrix
        assert cir.size() == 6


if __name__ == "__main__":
    unittest.main()
