import random
import unittest
import numpy as np
from numpy import size
from numpy.core._simd import targets
from QuICT.core import Qureg, Circuit, gate, Qubit
from QuICT.core.gate import *
from QuICT.core.utils import GateType
from QuICT.lib.qasm.node import Gate
from QuICT.core.gate.gate_builder import build_random_gate,build_gate


class TestGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Circuit unit test start!")
    @classmethod
    def tearDownClass(cls) -> None:
        print("The Circuit unit test finished!")

    def test_gate_build(self):
        cir = Circuit(10)
        # single qubit gate
        h1 = HGate()
        h1 | cir(1)  # 1
        H | cir  # 11

        # single qubit gate with param
        my_u1 = U1Gate([1])
        my_u1 | cir(2)  # 12
        U1(0) | cir(1)  # 13

        # two qubit gate
        my_CX = CX & [3, 4]
        my_CX | cir  # 14
        CX | cir([3, 4])  # 15

        # two qubit gate with param
        # my_CU3 = CU3 & [5,6]
        # my_CU3(1, 0, 0) | cir
        Rzz(1) | cir([1, 2])
        CU3(1, 0, 0) | cir([5, 6])  # 16

        # complexed gate
        CCRz(1) | cir([7, 8, 9])  # 17
        cg_ccrz = CCRz.build_gate()
        cg_ccrz | cir([7, 8, 9])  # 22

        assert len(cir.gates) == 23
        cir.draw()
        # print(cir.qasm())
    def test_gate_name(self):
        my_gate = HGate()
        my_gate.name.split('-')[0] == str(GateType.h)
        # single qubit gate
        q = Qureg(1)
        cir = Circuit(q)
        my_gate | cir(0)
        assert cir.gates[0].name.split('-')[1] == str(q[0].id[:6])
        assert cir.gates[0].name.split('-')[2] == str(0)
        # two qubit gate
        my_gate = CZGate()
        assert my_gate.name.split('-')[0] == str(GateType.cz)
        q= Qureg(2)
        cir = Circuit(q)
        my_gate | cir([0,1])
        print(cir.gates[0].name)
        assert cir.gates[0].name.split('-')[1] == str(q[0].id[:6])
        assert cir.gates[0].name.split('-')[2] == str(0)

    def test_gate_attribute(self):
        # test single gate
        assert H.is_single() and not CRz.is_single()
        assert Y.is_single() and not CZ.is_single()

        # test control single
        assert CRz.is_control_single() and not H.is_control_single()
        assert CZ.is_control_single() and not U1.is_control_single()

        # test Clifford gate
        assert S.is_clifford() and not T.is_clifford()
        assert CX.is_clifford() and not T.is_clifford()

        # test diagonal gate
        assert S.is_diagonal() and not H.is_diagonal()
        assert CZ.is_diagonal() and not Y.is_diagonal()

        # test Pauli gate
        assert X.is_pauli() and not Rx.is_pauli()

        # test unitary gate diagonal
        amatrix = np.array([
            [1, 0],
            [0, 1j]
        ], dtype=np.complex128)
        bmatrix = np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)]
        ], dtype=np.complex128)
        aunitary = Unitary(amatrix)
        bunitary = Unitary(bmatrix)
        assert aunitary.is_diagonal() and not bunitary.is_diagonal()

        # test matrix_type
        assert S.matrix_type != H.matrix_type
        assert S.matrix_type != CX.matrix_type


    # test special gate
    assert Measure.is_special() and not H.is_special()

    # # test matrix_type
    # def test_matrix_type(self):
    #     assert S.matrix_type and not X.matrix_type

    #gate_builder_test
    def test_build_gate(self):
        for _ in range(10):
            typelist_1qubit = [GateType.rx, GateType.ry, GateType.rz]
            typelist_2qubit = [
                GateType.cx, GateType.cy, GateType.crz,
                GateType.ch, GateType.cz, GateType.Rxx,
                GateType.Ryy, GateType.Rzz, GateType.fsim
            ]
            # build 1qubit gate
            gate_type = GateType.h
            q6 = Qureg(1)
            g6 = build_gate(gate_type, q6)
            assert g6.type == gate_type and g6.assigned_qubits == q6

            # build 1qubit gate with params
            gate_type = typelist_1qubit[random.randint(0, len(typelist_1qubit) - 1)]
            q1 = Qureg(1)
            params = [random.random()]
            g1 = build_gate(gate_type, q1, params)
            assert g1.type == gate_type and g1.assigned_qubits == q1

            # build 2qubits gate
            gate_type = typelist_2qubit[random.randint(0, len(typelist_2qubit) - 1)]
            q2 = Qureg(2)
            g2 = build_gate(gate_type, q2)
            assert g2.type == gate_type and g2.assigned_qubits == q2

            # build 2qubits gate with params
            gate_type = GateType.cu3
            q3 = Qureg(2)
            params = [1, 1, 1]
            g3 = build_gate(gate_type, q3, params)
            assert g3.pargs == params and g3.assigned_qubits == q3

            # build unitary gate
            from scipy.stats import unitary_group

            gate_type = GateType.unitary
            matrix = unitary_group.rvs(2 ** 3)
            g4 = build_gate(gate_type, [1, 2, 3], matrix)
            assert g4.matrix.shape == (8, 8)

            # build special gate
            gate_type = GateType.measure
            q5 = Qubit()
            g5 = build_gate(gate_type, q5)
            assert g5.assigned_qubits[0] == q5

    def test_build_random_gate(self):
            for _ in range(10):
                typelist_1qubit = [GateType.rx, GateType.ry, GateType.rz]
                typelist_2qubit = [
                    GateType.cx, GateType.cy, GateType.crz,
                    GateType.ch, GateType.cz, GateType.Rxx,
                    GateType.Ryy, GateType.Rzz, GateType.fsim
                ]
                # build random 1qubit gate
                gate_type = typelist_1qubit[random.randint(0, len(typelist_1qubit) - 1)]
                rg1 = build_random_gate(gate_type, 10, random_params=True)
                assert rg1.type == gate_type

                # build random 2qubits gate
                gate_type = typelist_2qubit[random.randint(0, len(typelist_2qubit) - 1)]
                rg2 = build_random_gate(gate_type, 10, random_params=True)
                assert rg2.type == gate_type

if __name__ == "__main__":
    unittest.TestCase()






















    # def __int__(self):
    #     self.gates = 10
    #     self.targets = 5
    #     self.controls = 5
    #     self.basicgate = BasicGate(self.gates)

    # def test_gate_attr(self):
    #     gate = TestGate.based_gate
    #     compositeGate = gate.gates
    #     self_controls = set(self.cargs)
    #     self_targets = set(self.targs)
    #
    # def test_gate_is_single(self):
    #     is_single.append()
    #     assert self.targets + self.controls == 1
    #
    #
    #     # gate.cargs = [targets[carg] for carg in gate.cargs]
    #     # gate.targs = [targets[targ] for targ in gate.targs]
    #
    #     # assert size(gate.cargs) == size(gate.targs)
    #
    #     # #append gate by different type
    #     # S | cir(qureg[0])
    #     # S | cir(0)
    #     # S & 0 | cir
    #     #
    #     # # check gate whether repeat
    #     # CX | cir([0,1,1])
    #
    #
    #     assert True
    # def test_gate_len(self):
    #     args_number = gate.controls + gate.targets
    #     assert len(args_number) == len(gate.controls+gate.targets)



if __name__=="__main__":
    unittest.TestCase()
