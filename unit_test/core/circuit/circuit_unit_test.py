import unittest

from QuICT.core import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.core.noise.noise_error import DampingError
from QuICT.core.operator.noise_gate import NoiseGate
from QuICT.core.operator.trigger import Trigger
from QuICT.core.qubit.qubit import Qureg
from QuICT.core.utils.gate_type import GateType
from QuICT.core.gate import *


class TestCircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Circuit unit test start!")
        cls.qubits = 10
        cls.based_circuit = Circuit(cls.qubits)

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Circuit unit test finished!")
        del cls.based_circuit

    def test_circuit_build(self):
        cir = TestCircuit.based_circuit
        qureg = cir.qubits
        # single qubit gate
        H | cir(0)

        cgate_single = H
        cgate_single | cir(1)

        cgate_single2 = gate_builder(GateType.h) & Qureg(1)
        cgate_single2 | cir(2)

        assert len(cir.gates) == 3

        # single qubit gate with param
        U1(np.pi / 2) | cir(1)

        cgate_single = U1
        cgate_single(0) | cir(2)

        cgate_single2 = gate_builder(GateType.u1, params=[(np.pi / 2)])
        cgate_single2 & 3 | cir
        assert len(cir.gates) == 6

        # two qubit gate
        CX | cir([1, 2])

        cgate_double = CX & [3, 4]
        cgate_double | cir

        cgate_double2 = gate_builder(GateType.cx) & [5, 6]
        cgate_double2 | cir
        assert len(cir.gates) == 9

        # two qubit gate with param
        Rzz(np.pi / 2) | cir([1, 2])
        CU3(np.pi / 2, 0, 0) | cir([3, 4])

        cgate_double = Ryy & [5, 6]
        cgate_double(np.pi / 2) | cir

        cgate_double2 = gate_builder(GateType.cu3, random_params=True) & [7, 8]
        cgate_double2 | cir

        cgate_double3 = Rzz.build_gate()
        cgate_double3 | cir([7, 8])
        assert len(cir.gates) == 14

        # complexed gate
        CCRz(np.pi / 2) | cir([1, 2, 3])

        cgate_complex = CCX & [4, 5, 6]
        cgate_complex | cir

        cgate_complex2 = gate_builder(GateType.ccrz, random_params=True)
        cgate_complex2 | cir([7, 8, 9])

        cgate_complex3 = CCRz.build_gate()
        cgate_complex3(np.pi / 2) | cir([2, 4, 6])

        matrix = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.complex128)
        cgate = Unitary(matrix)
        cgate_complex4 = cgate.build_gate()
        cgate_complex4 | cir(0)
        assert len(cir.gates) == 19

        # append composite gate
        qft_gate = QFT(3)
        qft_gate | cir([0, 1, 3])

        cgate_append = CompositeGate()
        CCRz(1) | cgate_append([0, 1, 2])
        cgate_append | cir
        assert len(cir.gates) == 21

        cgate_complex3 | cgate_append
        cgate_complex3 | cir
        c_1 = cir.gates[-1]

        cgate_complex3 ^ cgate_append
        cgate_append | cir

        c_2 = cir.gates[-1]
        assert c_1 != c_2
        assert len(cir.gates) == 23

        cir_unitary = Circuit(5)
        cgate = CompositeGate()
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.complex128)
        cgate_complex4 = Unitary(matrix, MatrixType.identity)
        com = cgate_complex4.build_gate()
        com | cgate
        cgate | cir_unitary

        cir_unitary2 = Circuit(5)
        cgate1 = CompositeGate()
        com ^ cgate1
        cgate1 | cir_unitary2
        assert cir_unitary.qasm() == cir_unitary2.inverse().qasm()

        # append gate by qubits/qureg
        S | cir(qureg[0])
        CX | cir(qureg[1, 3])
        assert len(cir.gates) == 25

        # append cir to cir
        cir1 = Circuit(5)
        H | cir1
        cir2 = Circuit(3)
        cir1 | cir2
        assert cir2.width() == 8 and cir2.size() == 5

        qureg = Qureg(3)
        cir1 = Circuit(qureg)
        cir2 = Circuit(qureg)
        cir1 | cir2
        assert cir2.width() == 3

        cir1 = Circuit(5)
        H | cir1(3)
        cir2 = Circuit(2)
        cir1([2, 3]) | cir2
        assert cir1.width() == 5 and cir1.gates[-1].targs == [3]

        # random append gates to circuit
        build_gate_typelist = [GateType.ccrz, GateType.cswap, GateType.ccz, GateType.ccx]
        prob = [0.25, 0.25, 0.25, 0.25]
        cir1 = Circuit(TestCircuit.qubits)
        cir1.random_append(10)
        cir1.random_append(10, build_gate_typelist)
        cir1.random_append(10, build_gate_typelist, random_params=True)
        cir1.random_append(10, build_gate_typelist, random_params=True, probabilities=prob)
        assert 1

        # append supremacy circuit
        cir.supremacy_append()
        cir.supremacy_append(repeat=3, pattern="ABCD")
        assert 1

        # append sub circuit
        cir = TestCircuit.based_circuit
        sub_cir_without_remove = cir.sub_circuit(qubit_limit=[0, 1, 2])
        assert cir.size() == 351
        assert sub_cir_without_remove.width() == 3

        sub_cir_with_remove = cir.sub_circuit(max_size=120, qubit_limit=[0, 3])
        assert cir.size() + sub_cir_with_remove.size() == 412
        assert sub_cir_without_remove.width() == 3

        # transform circuit to composite gate
        cir_trans = Circuit(5)
        cir_trans.random_append(10)
        composite_gate = cir_trans.to_compositegate()
        cir_trans2 = Circuit(5)
        cir_trans2.extend(composite_gate)
        assert cir_trans.size() == cir_trans2.size()

        # add qubits to circuit with ancillary_qubit or not
        cir_trans.add_qubit(5, is_ancillary_qubit=False)
        assert cir_trans.width() == 10
        cir_trans.add_qubit(5, is_ancillary_qubit=True)
        assert cir_trans.width() == 15
        assert cir_trans.ancilla_qubits == [10, 11, 12, 13, 14]

        # reset qubits to circuit
        cir.reset_qubits
        assert cir.width() == 10

        # extend gate/circuit to circuit
        cir_extend = Circuit(3)
        cir_trans.extend(cir_extend)    # extend a empty circuit
        assert cir_trans.width() == 18 and cir_trans.size() == 10

        cir_extend.extend(cir_extend)   # extend same circuit
        assert cir_extend.width() == 3 and cir_extend.size() == 0

        cir_extend.extend(cir_trans)    # empty extend circuit
        assert cir_extend.width() == 18 and cir_extend.size() == 10

        cir_extend = Circuit(5)
        cir_extend.random_append(10)
        cir_trans.extend(cir_extend)    # extend random circuit
        assert cir_trans.width() == 23 and cir_trans.size() == 20

        cgate = CompositeGate()
        H | cgate(1)
        CX | cgate([1, 2])
        cir2.extend(cgate)  # extend compositegate
        assert cir_extend.width() == 5 and cir_extend.size() == 10

        cgate2 = CompositeGate()
        cir_extend.extend(cgate2)   # extend empty compositegate
        assert cir_extend.width() == 5 and cir_extend.size() == 10

        # append gate to circuit
        cir3 = Circuit(3)
        cir3.append(H)
        assert cir3.size() == 3

        # insert gate to circuit
        cir_insert = CompositeGate()
        H | cir_insert(0)
        cir3.insert(cir_insert, 1)
        cir_insert | cir3

        cir_insert2 = CX & [1, 2]
        cir3.insert(cir_insert2, 0)
        assert cir3.gates[0] != cir3.gates[1]

        # inverse circuit
        cir_inverse = cir3.inverse()
        assert cir_inverse.size() == cir3.size() and cir_inverse.qasm() != cir3.qasm()

    def test_circuit_info(self):
        # add all type of gates to circuit
        cir = Circuit(wires=10, name="cir", ancilla_qubits=[3, 4])

        H | cir(0)
        U1(np.pi / 2) | cir(1)
        CX | cir([1, 2])
        Rzz(np.pi / 2) | cir([1, 2])
        CU3(np.pi / 2, 0, 0) | cir([3, 4])
        CCZ | cir([1, 2, 3])

        qureg = cir.qubits
        S | cir(qureg[0])
        CY | cir(qureg[1, 3])
        CSwap | cir(qureg[3, 4, 5])

        c1 = Hy
        c1 | cir(1)
        c1 = U2
        c1(0, 0) | cir(2)
        c1 = CX & [3, 4]
        c1 | cir
        c1 = Ryy & [5, 6]
        c1(np.pi / 4) | cir
        c1 = CCX & [4, 5, 6]
        c1 | cir

        c2 = gate_builder(GateType.sdg) & Qureg(1)
        c2 | cir(2)
        c2 = gate_builder(GateType.rx, params=[(np.pi / 2)])
        c2 & 3 | cir
        c2 = gate_builder(GateType.cx) & [5, 6]
        c2 | cir
        c2 = gate_builder(GateType.cu3, random_params=True) & [7, 8]
        c2 | cir
        c2 = gate_builder(GateType.ccrz, random_params=True)
        c2 | cir([7, 8, 9])

        c3 = Rxx.build_gate()
        c3 | cir([7, 8])

        qft_gate = QFT(3)
        qft_gate | cir([0, 1, 3])

        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.complex128)
        cgate_complex4 = Unitary(matrix, MatrixType.identity)
        u_gate = cgate_complex4.build_gate()
        u_gate | cir

        qureg = Qureg(5)
        cir1 = Circuit(qureg)
        SX | cir1(1)
        SY | cir1(1)
        SW | cir1(1)
        ID | cir1(4)
        T | cir1(3)
        T_dagger | cir1(3)
        Phase(0) | cir1(3)
        GPhase(0) | cir1(4)
        CZ | cir1([0, 1])
        CH | cir1([0, 1])
        CRz(0) | cir1([4, 3])
        Measure | cir1(2)
        Reset | cir1(1)
        Barrier | cir1(2)
        iSwap_dagger | cir1([0, 1])
        sqiSwap | cir1([0, 1])
        FSim(np.pi / 2, 0) | cir1([3, 4])
        cir1 | cir

        cir.qasm()
        assert 1

        cgate0 = CompositeGate()
        X | cgate0(2)
        cgate1 = CompositeGate()
        X | cgate1(2)
        X | cgate1(2)
        gates = [
            cgate0,
            cgate1
        ]

        trigger = Trigger(1, gates, "trigger")
        trigger | cir([0])

        amp_err = DampingError(amplitude_prob=0.2, phase_prob=0, dissipation_state=0.3)
        noise = NoiseGate(Z, noise=amp_err)
        noise | cir([5])

        assert cir.width() == 15
        assert cir.size() == 61
        assert cir.depth() == 19
        assert len(cir.qubits) == 15
        assert cir.count_1qubit_gate() == 35
        assert cir.count_2qubit_gate() == 22
        assert cir.count_gate_by_gatetype(GateType.measure) == 1
        assert cir.name == "cir" and cir.ancilla_qubits == [3, 4]

    def test_circuit_decomposition(self):
        build_gate_typelist = [GateType.ccrz, GateType.cswap, GateType.ccz, GateType.ccx]
        cir = Circuit(TestCircuit.qubits)
        cir.random_append(10, build_gate_typelist)
        QFT(5) | cir
        cir.gate_decomposition()
        for gate in cir.gates:
            assert gate.controls + gate.targets < 3

    def test_circuit_matrix(self):
        from QuICT.simulation.state_vector import StateVectorSimulator
        from QuICT.simulation.unitary import UnitarySimulator

        cir = Circuit(TestCircuit.qubits)
        cir.random_append(10)
        state_vector_cir = StateVectorSimulator().run(cir)

        cir_matrix = cir.matrix()
        state_vector_matrix = UnitarySimulator().run(cir_matrix)
        assert np.allclose(state_vector_cir, state_vector_matrix)

        assert cir.matrix("GPU").dtype == np.complex128
        cir.set_precision("single")
        assert len(cir.flatten_gates()) == cir.size()
        assert cir.precision == "single"

        mp_gate = CZ & [1, 3]
        mp_data = mp_gate.expand(cir.width())
        assert mp_data.shape == (1 << 10, 1 << 10)


if __name__ == "__main__":
    unittest.main()
