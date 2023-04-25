import unittest
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.operator import *
from QuICT.simulation.state_vector import StateVectorSimulator


class TestOperator(unittest.TestCase):
    def test_checkpoint(self):
        # Build Circuit and CompositeGate
        cir = Circuit(5)
        QFT(5) | cir
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
        IQFT(5) | cir

        # add CompositeGate with cp_child into cir
        cp_child | cgate
        cp_child | cgate1

        cgate | cir
        cgate1 | cir

        # get target gate from circuit
        # x_gate = cir.gates[qft_size - 1]
        # y_gate = cir.gates[qft_size + 5]

        # assert x_gate.type == GateType.x and y_gate.type == GateType.y

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

        sim = StateVectorSimulator()
        sv = sim.run(cir)

        if cir[0].measured:
            assert np.isclose(sv[-1], 1)
        else:
            assert np.isclose(sv[0], 1)

    def test_noise(self):
        pass

if __name__ == "__main__":
    unittest.main()
