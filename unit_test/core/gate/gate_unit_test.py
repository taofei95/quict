import random
import unittest
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import *


class TestGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Gate unit test start!")

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Gate unit test finished!")

    def test_gate_attribute(self):
        # test single gate
        assert H.is_single() and not CRz.is_single()

        # test control single
        assert CRz.is_control_single() and not S.is_control_single()

        # test Clifford gate
        assert S.is_clifford() and not T.is_clifford()

        # test diagonal gate
        assert Rz.is_diagonal() and not H.is_diagonal()

        # test Pauli gate
        assert X.is_pauli() and not Rx.is_pauli()

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

        # test matrix_type
        assert S.matrix_type != CX.matrix_type
        assert GPhase.matrix_type == MatrixType.diagonal
        assert Hy.matrix_type == MatrixType.normal
        assert S_dagger.matrix_type == MatrixType.control
        assert X.matrix_type == MatrixType.swap
        assert Y.matrix_type == MatrixType.reverse
        assert Measure.matrix_type == MatrixType.special
        assert Rzz.matrix_type == MatrixType.diag_diag
        assert FSim.matrix_type == MatrixType.ctrl_normal
        assert Ryy.matrix_type == MatrixType.normal_normal
        assert Rzx.matrix_type == MatrixType.diag_normal

        assert H.get_matrix("single").dtype != H.get_matrix("double").dtype

        assert CX.controls == 1 and S.controls == 0 and CCRz.controls == 2
        assert Rxx.targets == 2 and Measure.targets == 1
        assert U1.qasm_name == "u1"
        assert iSwap_dagger.type == GateType.iswapdg

    def test_gate_inverse(self):
        single_ide = np.identity(2, np.complex128)
        # one qubit gate
        s_inv = S.inverse()
        assert np.allclose(np.dot(S.matrix, s_inv.matrix), single_ide)

        sdg_inv = S_dagger.inverse()
        assert np.allclose(np.dot(S_dagger.matrix, sdg_inv.matrix), single_ide)

        sy_inv = SY.inverse()
        assert np.allclose(np.dot(SY.matrix, sy_inv.matrix), single_ide)

        sw_inv = SW.inverse()
        assert np.allclose(np.dot(SW.matrix, sw_inv.matrix), single_ide)

        id_inv = ID.inverse()
        assert np.allclose(np.dot(ID.matrix, id_inv.matrix), single_ide)

        t_inv = T.inverse()
        assert np.allclose(np.dot(T.matrix, t_inv.matrix), single_ide)

        tdg_inv = T_dagger.inverse()
        assert np.allclose(np.dot(T_dagger.matrix, tdg_inv.matrix), single_ide)

        # Parameter Gate inverse test
        # 1-qubit gates
        alpha = np.random.random(1)[0] * np.pi
        u1 = U1(alpha)
        u1_inv = u1.inverse()
        assert np.allclose(np.dot(u1_inv.matrix, u1.matrix), single_ide)

        alpha = np.random.random(1)[0] * np.pi
        beta = np.random.random(1)[0] * np.pi
        u2 = U2(alpha, beta)
        u2_inv = u2.inverse()
        assert np.allclose(np.dot(u2_inv.matrix, u2.matrix), single_ide)

        alpha = np.random.random(1)[0] * np.pi
        beta = np.random.random(1)[0] * np.pi
        gamma = np.random.random(1)[0] * np.pi
        u3 = U3(alpha, beta, gamma)
        u3_inv = u3.inverse()
        assert np.allclose(np.dot(u3_inv.matrix, u3.matrix), single_ide)

        alpha = np.random.random(1)[0] * np.pi
        rx = Rx(alpha)
        rx_inv = rx.inverse()
        assert np.allclose(np.dot(rx_inv.matrix, rx.matrix), single_ide)

        alpha = np.random.random(1)[0] * np.pi
        ry = Ry(alpha)
        ry_inv = ry.inverse()
        assert np.allclose(np.dot(ry_inv.matrix, ry.matrix), single_ide)

        alpha = np.random.random(1)[0] * np.pi
        rz = Rz(alpha)
        rz_inv = rz.inverse()
        assert np.allclose(np.dot(rz_inv.matrix, rz.matrix), single_ide)

        alpha = np.random.random(1)[0]
        p = Phase(alpha)
        p_inv = p.inverse()
        assert np.allclose(np.dot(p_inv.matrix, p.matrix), single_ide)

        alpha = np.random.random(1)[0]
        gp = GPhase(alpha)
        gp_inv = gp.inverse()
        assert np.allclose(np.dot(gp_inv.matrix, gp.matrix), single_ide)

        # 2-qubit gates
        double_ide = np.identity(4)
        alpha = np.random.random(1)[0] * np.pi
        crz = CRz(alpha)
        crz_inv = crz.inverse()
        assert np.allclose(np.dot(crz_inv.matrix, crz.matrix), double_ide)

        alpha = np.random.random(1)[0] * np.pi
        cu1 = CU1(alpha)
        cu1_inv = cu1.inverse()
        assert np.allclose(np.dot(cu1_inv.matrix, cu1.matrix), double_ide)

        alpha = np.random.random(1)[0] * np.pi
        beta = np.random.random(1)[0] * np.pi
        gamma = np.random.random(1)[0] * np.pi
        cu3 = CU3(alpha, beta, gamma)
        cu3_inv = cu3.inverse()
        assert np.allclose(np.dot(cu3_inv.matrix, cu3.matrix), double_ide)

        alpha = np.random.random(1)[0] * np.pi
        beta = np.random.random(1)[0]
        fsim = FSim(alpha, beta)
        fsim_inv = fsim.inverse()
        assert np.allclose(np.dot(fsim_inv.matrix, fsim.matrix), double_ide)

        alpha = np.random.random(1)[0]
        rxx = Rxx(alpha)
        rxx_inv = rxx.inverse()
        assert np.allclose(np.dot(rxx_inv.matrix, rxx.matrix), double_ide)

        alpha = np.random.random(1)[0]
        ryy = Ryy(alpha)
        ryy_inv = ryy.inverse()
        assert np.allclose(np.dot(ryy_inv.matrix, ryy.matrix), double_ide)

        alpha = np.random.random(1)[0]
        rzz = Rzz(alpha)
        rzz_inv = rzz.inverse()
        assert np.allclose(np.dot(rzz_inv.matrix, rzz.matrix), double_ide)

        alpha = np.random.random(1)[0]
        rzx = Rzx(alpha)
        rzx_inv = rzx.inverse()
        assert np.allclose(np.dot(rzx_inv.matrix, rzx.matrix), double_ide)

        alpha = np.random.random(1)[0]
        ccrz = CCRz(alpha)
        ccrz_inv = ccrz.inverse()
        assert np.allclose(np.dot(ccrz_inv.matrix, ccrz.matrix), np.identity(8, np.complex128))

        # unitary gates
        from scipy.stats import unitary_group
        matrix = unitary_group.rvs(2 ** 2)
        u2 = Unitary(matrix).inverse()
        assert np.allclose(np.dot(matrix, u2.matrix), double_ide)

        # perm gates
        p = Perm(2, [0, 1])
        p_inverse = p.inverse()
        assert np.allclose(np.dot(p.matrix, p_inverse.matrix), p_inverse.matrix)

        p = Perm(3, [0, 1, 2])
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
            typelist_1qubit = [
                GateType.rx, GateType.ry, GateType.rz,
                GateType.hy, GateType.s, GateType.sdg,
                GateType.x, GateType.y, GateType.z,
                GateType.sx, GateType.sy, GateType.sw,
                GateType.id, GateType.u1, GateType.u2,
                GateType.u3, GateType.t, GateType.tdg,
                GateType.phase, GateType.gphase, GateType.measure,
                GateType.reset, GateType.barrier
            ]
            typelist_2qubit = [
                GateType.cx, GateType.cy, GateType.crz,
                GateType.ch, GateType.cz, GateType.rxx,
                GateType.ryy, GateType.rzz, GateType.fsim,
                GateType.cu1, GateType.cu3, GateType.swap,
                GateType.iswap, GateType.iswapdg, GateType.sqiswap
            ]
            typelist_3qubit = [GateType.ccx, GateType.ccz, GateType.ccrz, GateType.cswap]

            # build random 1qubit gate
            gate_type = typelist_1qubit[random.randint(0, len(typelist_1qubit) - 1)]
            rg1 = gate_builder(gate_type, random_params=True)
            assert rg1.type == gate_type

            # build random 2qubits gate
            gate_type = typelist_2qubit[random.randint(0, len(typelist_2qubit) - 1)]
            rg2 = gate_builder(gate_type, random_params=True)
            assert rg2.type == gate_type

            # build random 3qubits gate
            gate_type = typelist_3qubit[random.randint(0, len(typelist_3qubit) - 1)]
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
