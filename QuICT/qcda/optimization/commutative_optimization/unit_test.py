
import pytest

import numpy as np

from QuICT.algorithm import SyntheticalUnitary
from QuICT.core import *
from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization

def test_commute():
    a = CX & [0, 1]
    b = CX & [1, 0]
    print(a.commutative(b))

def test_parameterize():
    gates = CompositeGate()
    with gates:
        X & 0
        Y & 1
        Z & 2
        X & 0
    phase_angle = 0
    gates_para = CompositeGate()
    for gate in gates:
        gate_para, phase = CommutativeOptimization.parameterize(gate)
        gates_para.append(gate_para)
        phase_angle += phase
    with gates_para:
        Phase(phase_angle) & 0
    gates_para.print_information()
    assert np.allclose(gates.matrix(), gates_para.matrix())

def test_deparameterize():
    gates = CompositeGate()
    with gates:
        Rx(np.pi) & 0
        Rx(-np.pi) & 1
        Ry(np.pi) & 2
        Ry(-np.pi) & 3
        Rz(np.pi) & 4
        Rz(-np.pi) & 5
        Rz(np.pi/2) & 0
        Rz(-np.pi/2) & 1
        Rz(np.pi/4) & 2
        Rz(-np.pi/4) & 3
        Rx(np.pi/2) & 0
        Rx(5*np.pi/2) & 1
        Ry(np.pi/2) & 2
        Ry(5*np.pi/2) & 3
    phase_angle = 0
    gates_depara = CompositeGate()
    for gate in gates:
        gate_depara, phase = CommutativeOptimization.deparameterize(gate)
        gates_depara.append(gate_depara)
        phase_angle += phase
    with gates_depara:
        Phase(phase_angle) & 0
    gates_depara.print_information()
    assert np.allclose(gates.matrix(), gates_depara.matrix())

def test_combine():
    gate_x = X & 0
    gate_y = X & 0
    gate = CommutativeOptimization.combine(gate_x, gate_y)
    gates = CompositeGate()
    with gates:
        gate & gate.targs
    gates.print_information()

# Be aware that too many types at the same time may not benefit to the test,
# unless the size of the random circuit is also large.
typelist = [GATE_ID['Rx'], GATE_ID['Ry'], GATE_ID['Rz'],
            GATE_ID['X'], GATE_ID['Y'], GATE_ID['Z'],
            GATE_ID['S'], GATE_ID['T'], GATE_ID['H'],
            GATE_ID['CX'], GATE_ID['CRz'], GATE_ID['FSim']]
typelist = [GATE_ID['Rx'], GATE_ID['Ry'], GATE_ID['Rz'], GATE_ID['X'], GATE_ID['Y'], GATE_ID['Z'], GATE_ID['CX']]
# typelist = [GATE_ID['Rx'], GATE_ID['Ry'], GATE_ID['Rz']]
# typelist = [GATE_ID['X'], GATE_ID['Y'], GATE_ID['Z']]
# typelist = [GATE_ID['CX'], GATE_ID['CRz'], GATE_ID['FSim']]
# typelist = [GATE_ID['U2'], GATE_ID['U3'], GATE_ID['CU3']]

def test():
    for _ in range(100):
        n = 5
        circuit = Circuit(n)
        circuit.random_append(rand_size=100, typeList=typelist)
        # circuit.print_information()
        # circuit.draw()

        gates = CommutativeOptimization.execute(circuit)
        circuit_opt = Circuit(n)
        circuit_opt.set_exec_gates(gates)

        # circuit_opt.print_information()
        # circuit_opt.draw()

        original = SyntheticalUnitary.run(circuit)
        opt = SyntheticalUnitary.run(circuit_opt)
        phase = opt.dot(np.linalg.inv(original))
        assert np.allclose(original, opt)
        # assert np.allclose(phase, phase[0, 0] * np.eye(2 ** n), rtol=1e-10, atol=1e-10)

if __name__ == '__main__':
    # test()
    pytest.main(["./unit_test.py"])
