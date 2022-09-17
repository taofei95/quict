import unittest
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.noise.utils import NoiseChannel
from QuICT.core.operator import *
from QuICT.core.noise import BitflipError
from QuICT.simulation.state_vector import CircuitSimulator, ConstantStateVectorSimulator


class TestOperator(unittest.TestCase):
    def test_checkpoint(self):
        # Build Circuit and CompositeGate
        cir = Circuit(5)
        QFT.build_gate(5) | cir
        qft_size = cir.size()

        cgate = CompositeGate()
        X | cgate(0)
        X | cgate(1)
        cgate1 = CompositeGate()
        Y | cgate1(0)
        Y | cgate1(1)

        # Initial CheckPoint and CheckPointChild
        cp = CheckPoint()
        cp_child = cp.get_child(5)

        # add CheckPoint into Circuit
        cp | cir
        IQFT.build_gate(5) | cir

        # add CompositeGate with cp_child into cir
        cp_child | cgate
        cp_child | cgate1

        cgate | cir
        cgate1 | cir

        # get target gate from circuit
        x_gate = cir.gates[qft_size]
        y_gate = cir.gates[qft_size + 5]

        assert x_gate.type == GateType.x and y_gate.type == GateType.y

    def test_noisegate(self):
        # Build noise gate
        based_gate = H & 3
        error = BitflipError(0.1)
        noise_gate = NoiseGate(based_gate, error)

        # Test attribute
        assert len(noise_gate.cargs) == 0
        assert noise_gate.targs == [3]
        assert noise_gate.type == GateType.h
        assert noise_gate.noise_type == NoiseChannel.pauil

        # Test noise kraus matrix
        noise_matrixs = noise_gate.noise_matrix
        I_noise_error = np.sqrt(1 - 0.1) * ID.matrix
        X_noise_error = np.sqrt(0.1) * X.matrix

        assert np.allclose(noise_matrixs[0], np.dot(I_noise_error, H.matrix))
        assert np.allclose(noise_matrixs[1], np.dot(X_noise_error, H.matrix))

    def test_trigger(self):
        # Build circuit and compositegate
        cir = Circuit(4)
        H | cir(0)

        cgate = CompositeGate()
        CX | cgate([0, 1])
        CX | cgate([1, 2])
        CX | cgate([2, 3])

        # Build Trigger
        trigger = Trigger(1, [None, cgate])
        trigger | cir(0)

        sim = CircuitSimulator()
        sv = sim.run(cir)

        if cir[0].measured:
            assert np.isclose(sv[-1], 1)
        else:
            assert np.isclose(sv[0], 1)


if __name__ == "__main__":
    unittest.main()
