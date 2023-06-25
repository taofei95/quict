import random
import unittest
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import *


class TestGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Gate unit test start!")
        cls.default_1_qubits_gate = [
            GateType.h, GateType.hy, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z,
            GateType.sx, GateType.sy, GateType.sw, GateType.id, GateType.u1, GateType.u2, GateType.u3,
            GateType.rx, GateType.ry, GateType.rz, GateType.t, GateType.tdg, GateType.phase, GateType.gphase,
            GateType.measure, GateType.reset, GateType.barrier
        ]
        cls.default_2_qubits_gate = [
            GateType.cx, GateType.cz, GateType.ch, GateType.crz, GateType.cu1, GateType.cu3, GateType.fsim,
            GateType.rxx, GateType.ryy, GateType.rzz, GateType.swap, GateType.iswap, GateType.iswapdg, GateType.sqiswap
        ]
        cls.default_3_qubits_gate = [GateType.ccx, GateType.ccz, GateType.ccrz, GateType.cswap]

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Gate unit test finished!")

    def test_gate_attribute(self):
        single_qubit_gate =[
            H, Hy, S, S_dagger, X, Y, Z, SX, SY, SW, ID, U1, U2, U3,
            Rx, Ry, Rz, T, T_dagger, Phase, GPhase, Measure, Reset, Barrier
        ]
        single_control_gate = [CZ, CX, CY, CRz, CU1, CU3]
        clifford_gate = [X, Y, Z, H, S, S_dagger, CX]
        diagonal_matrix_gate = [Rz, GPhase, CRz, CCRz]
        pauli_gate = [X, Y, Z, ID]
        special_matrix_gate = [Measure, Reset, Barrier]
        complex_matrix_gate = [Rzz, FSim, Ryy, Rzx]
        matrix_type_list = [
            single_qubit_gate, single_control_gate, clifford_gate,
            diagonal_matrix_gate, pauli_gate, special_matrix_gate
            ]

        for _ in range(10):
            for gate_index in range(len(matrix_type_list)):
                gate = random.choice(matrix_type_list[gate_index])
                assert gate.matrix_type != random.choice(complex_matrix_gate).matrix_type
                if gate_index == 0:
                    assert gate.is_single()
                elif gate_index == 1:
                    assert gate.is_control_single()
                elif gate_index == 2:
                    assert gate.is_clifford()
                elif gate_index == 3:
                    assert gate.is_diagonal() and gate.matrix_type == MatrixType.diagonal
                elif gate_index == 4:
                    assert gate.is_pauli()
                elif gate_index == 5:
                    assert gate.is_special() and gate.matrix_type == MatrixType.special

        # test special gate
        assert Measure.is_special() and not X.is_special()

        # test identity gate
        assert ID.is_identity() and not H.is_identity()

        # test unitary gate diagonal
        amatrix = np.array([
            [1j, 0],
            [0, 1]
        ], dtype=np.complex128)
        bmatrix = np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)]
        ], dtype=np.complex128)
        aunitary = Unitary(amatrix)
        bunitary = Unitary(bmatrix)
        assert aunitary.is_diagonal() and not bunitary.is_diagonal()

    def test_gate_inverse(self):
        gate_list = [
            GateType.u1, GateType.rx, GateType.ry, GateType.phase, GateType.gphase,
            GateType.cu1, GateType.rxx, GateType.ryy, GateType.rzz, GateType.rzx,
            GateType.rz, GateType.crz, GateType.ccrz, GateType.fsim, GateType.s,
            GateType.sdg, GateType.sy, GateType.sw, GateType.t, GateType.tdg
        ]
        # normal gates
        for _ in range(10):
            gate = gate_builder(random.choice(gate_list), random_params=True)
            gate_inv = gate.inverse()
            qidxes = gate.controls + gate.targets
            gate_ide = np.identity(2 ** qidxes, np.complex128)
            assert np.allclose(np.dot(gate.matrix, gate_inv.matrix), gate_ide)

        # unitary gates
        from scipy.stats import unitary_group
        matrix = unitary_group.rvs(2 ** 2)
        u2 = Unitary(matrix).inverse()
        double_ide = np.identity(4, np.complex128)
        assert np.allclose(np.dot(matrix, u2.matrix), double_ide)

        # perm gates
        for target_num in range(5):
            p = Perm(target_num, list(range(target_num)))
            p_inverse = p.inverse()
            assert np.allclose(np.dot(p.matrix, p_inverse.matrix), p_inverse.matrix)

    def test_gate_commutative(self):
        cir = Circuit(5)
        cgate = CX & [0, 1]
        cgate | cir
        cgate2 = U2(1, 0)
        cgate2 | cir
        cgate2.commutative(cgate)
        assert True

    def test_build_random_gate(self):
        for _ in range(10):
            # build random 1qubit gate
            gate_type = self.default_1_qubits_gate[random.randint(0, len(self.default_1_qubits_gate) - 1)]
            rg1 = gate_builder(gate_type, random_params=True)
            assert rg1.type == gate_type

            # build random 2qubits gate
            gate_type = self.default_2_qubits_gate[random.randint(0, len(self.default_2_qubits_gate) - 1)]
            rg2 = gate_builder(gate_type, random_params=True)
            assert rg2.type == gate_type

            # build random 3qubits gate
            gate_type = self.default_3_qubits_gate[random.randint(0, len(self.default_3_qubits_gate) - 1)]
            rg3 = gate_builder(gate_type, random_params=True)
            assert rg3.type == gate_type

    def test_gate_expand(self):
        # single qubit quantum gate expand test
        single_gate = H
        expand_sgate1 = single_gate.expand(3)
        expand_sgate2 = single_gate.expand([0, 1, 2])

        cir = Circuit(3)
        H | cir(0)
        assert np.allclose(expand_sgate1, cir.matrix()) and np.allclose(expand_sgate2, cir.matrix())

        # single qubit assigned quantum gate expand test
        single_gate_assigned = H & 1
        expand_sagate1 = single_gate_assigned.expand(3)
        expand_sagate2 = single_gate_assigned.expand([0, 1, 2])

        cir = Circuit(3)
        H | cir(1)
        assert np.allclose(expand_sagate1, cir.matrix()) and np.allclose(expand_sagate2, cir.matrix())

        # double-qubits quantum gate expand test
        double_gate = CX
        expand_dgate1 = double_gate.expand(3)
        expand_dgate2 = double_gate.expand([0, 1, 2])

        cir = Circuit(3)
        CX | cir([0, 1])
        assert np.allclose(expand_dgate1, cir.matrix()) and np.allclose(expand_dgate2, cir.matrix())

        # double-qubits assigned quantum gate expand test
        double_gate_assigned = CX & [1, 2]
        expand_sdgate1 = double_gate_assigned.expand(3)
        expand_sdgate2 = double_gate_assigned.expand([0, 1, 2])

        cir = Circuit(3)
        CX | cir([1, 2])
        assert np.allclose(expand_sdgate1, cir.matrix()) and np.allclose(expand_sdgate2, cir.matrix())


if __name__ == "__main__":
    unittest.main()
