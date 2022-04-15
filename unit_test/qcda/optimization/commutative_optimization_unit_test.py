
import pytest

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization


def test_parameterize():
    test_list = [X, Y, Z, SX, SY, S, S_dagger, T, T_dagger]
    for gate in test_list:
        gate = gate & 0
        gate_para, phase = CommutativeOptimization.parameterize(gate)
        assert np.allclose(gate.matrix, np.exp(1j * phase) * gate_para.matrix)


def test_deparameterize():
    # Rx
    for k in range(8):
        gate = Rx(k * np.pi / 2) & 0
        gates_depara, phase = CommutativeOptimization.deparameterize(gate)
        assert np.allclose(gate.matrix, np.exp(1j * phase) * gates_depara.matrix())
    # Ry
    for k in range(8):
        gate = Ry(k * np.pi / 2) & 0
        gates_depara, phase = CommutativeOptimization.deparameterize(gate)
        assert np.allclose(gate.matrix, np.exp(1j * phase) * gates_depara.matrix())
    # Rz
    for k in range(16):
        gate = Rz(k * np.pi / 4) & 0
        gates_depara, phase = CommutativeOptimization.deparameterize(gate)
        assert np.allclose(gate.matrix, np.exp(1j * phase) * gates_depara.matrix())


# Be aware that too many types at the same time may not benefit to the test,
# unless the size of the random circuit is also large.
typelist = [GateType.rx, GateType.ry, GateType.rz,
            GateType.x, GateType.y, GateType.z,
            GateType.s, GateType.t, GateType.h,
            GateType.cx, GateType.crz, GateType.fsim]
typelist = [GateType.rx, GateType.ry, GateType.rz, GateType.x, GateType.y, GateType.z, GateType.cx]
typelist = [GateType.cx, GateType.h, GateType.s, GateType.t, GateType.x, GateType.y, GateType.z]
# typelist = [GateType.rx, GateType.ry, GateType.rz]
# typelist = [GateType.x, GateType.y, GateType.z]
# typelist = [GateType.cx, GateType.crz, GateType.fsim]
# typelist = [GateType.u2, GateType.u3, GateType.cu3]


def test():
    for _ in range(100):
        n = 5
        circuit = Circuit(n)
        circuit.random_append(rand_size=100, typelist=typelist)
        # print(circuit)
        # circuit.draw()

        gates = CommutativeOptimization.execute(circuit, deparameterization=True)
        circuit_opt = Circuit(n)
        circuit_opt.extend(gates)

        # print(circuit_opt)
        # circuit_opt.draw()

        original = SyntheticalUnitary.run(circuit)
        opt = SyntheticalUnitary.run(circuit_opt)
        # phase = opt.dot(np.linalg.inv(original))
        assert np.allclose(original, opt)
        # assert np.allclose(phase, phase[0, 0] * np.eye(2 ** n), rtol=1e-10, atol=1e-10)


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])