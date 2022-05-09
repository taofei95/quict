#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/12 11:49
# @Author  : Han Yu
# @File    : unit_test.py

from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.synthesis.mct import MCTOneAux, MCTLinearHalfDirtyAux, MCTLinearOneDirtyAux
from QuICT.algorithm import SyntheticalUnitary
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


def set_ones(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    gates = CompositeGate()
    with gates:
        n = len(qreg)
        for i in range(n):
            if N % 2 == 1:
                X & qreg[n - 1 - i]
            N = N // 2
    return gates


def test_MCT_Linear_Simulation_Half():
    max_qubit = 11
    for n in range(3, max_qubit):
        for m in range(1, int(np.ceil(n / 2))):
            circuit = Circuit(n)
            MCT = MCTLinearHalfDirtyAux()
            MCT.execute(m, n) | circuit
            unitary = SyntheticalUnitary.run(circuit)
            mat_mct = np.eye(1 << n)
            mat_mct[-1 << n - m:, -1 << n - m:] = \
                np.kron(np.eye(1 << n - m - 1), X.matrix.real)
            assert np.allclose(mat_mct, unitary)


def test_MCT_Linear_Simulation_One_functional():
    simulator = ConstantStateVectorSimulator()
    for n in range(3, 7):
        for control_bits in range(0, 1 << n - 2):
            circuit = Circuit(n)
            aux_idx = [0]
            controls_idx = [i for i in range(1, n - 1)]
            target_idx = [n - 1]
            # aux = circuit[aux_idx]
            controls = circuit[controls_idx]
            target = circuit[target_idx]

            set_ones(controls_idx, control_bits) | circuit
            # print("%d bits control = %d" % (n - 2, control_bits))
            MCT = MCTLinearOneDirtyAux()
            gates = MCT.execute(n)
            gates | circuit(controls_idx + target_idx + aux_idx)
            Measure | circuit
            simulator.run(circuit)
            if (
                (control_bits == 2 ** (n - 2) - 1 and int(target) == 0) or
                (control_bits != 2 ** (n - 2) - 1 and int(target) == 1) or
                # (int(aux) != 0) or
                (int(controls) != control_bits)
            ):
                print("when control bits are %d, the target is %d" % (control_bits, int(target)))
                assert 0


def test_MCT_Linear_Simulation_One_unitary():
    for n in range(3, 11):
        circuit = Circuit(n)
        aux_idx = [0]
        controls_idx = [i for i in range(1, n - 1)]
        target_idx = [n - 1]
        # aux = circuit[aux_idx]
        # controls = circuit[controls_idx]
        # target = circuit[target_idx]
        MCT = MCTLinearOneDirtyAux()
        gates = MCT.execute(n)
        gates | circuit(controls_idx + target_idx + aux_idx)
        unitary = SyntheticalUnitary.run(circuit)
        mat_mct = np.eye(1 << n - 1)
        mat_mct[(1 << n - 1) - 2:, (1 << n - 1) - 2:] = X.matrix.real
        mat_mct = np.kron(np.eye(2), mat_mct)
        assert np.allclose(mat_mct, unitary)


def test_MCTOneAux():
    for n in range(3, 11):
        circuit = Circuit(n)
        MCT = MCTOneAux()
        MCT.execute(n) | circuit
        unitary = SyntheticalUnitary.run(circuit)
        mat_mct = np.eye(1 << n - 1)
        mat_mct[(1 << n - 1) - 2:, (1 << n - 1) - 2:] = X.matrix.real
        mat_mct = np.kron(mat_mct, np.eye(2))
        assert np.allclose(mat_mct, unitary)
