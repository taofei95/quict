import unittest

from QuICT.core import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.core.noise import *
from QuICT.core.operator import NoiseGate
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

        cgate1 = H
        cgate1 | cir(1)

        cgate2 = gate_builder(GateType.h) & Qureg(1)
        cgate2 | cir
        assert len(cir.gates) == 3

        # single qubit gate with param
        U1(np.pi / 2) | cir(1)

        cgate1 = U1
        cgate1 | cir(2)

        cgate2 = gate_builder(GateType.u1, params=[(np.pi / 2)]) & 3
        cgate2 | cir
        assert len(cir.gates) == 6

        # two qubit gate
        CX | cir([1, 2])
        
        cgate1 = CX & [3, 4]
        cgate1 | cir
        
        cgate2 = gate_builder(GateType.cx) & [5, 6]
        cgate2 | cir
        assert len(cir.gates) == 9

        # two qubit gate with param
        Rzz(np.pi / 2) | cir([1, 2])
        CU3(np.pi / 2, 0, 0) | cir([3, 4])
        
        cgate1 = Ryy & [5, 6]
        cgate1(np.pi / 2) | cir

        cgate2 = gate_builder(GateType.cu3, random_params=True) & [7, 8]
        cgate2 | cir

        cgate3 = Rzz.build_gate()
        cgate3 | cir([7, 8])
        assert len(cir.gates) == 14
        
        # complexed gate
        CCRz(np.pi / 2) | cir([1, 2, 3])

        cgate1 = CCX & [4, 5, 6]
        cgate1 | cir

        cgate2 = gate_builder(GateType.ccrz, random_params=True)
        cgate2 | cir([7, 8, 9])
        
        cgate3 = CCRz.build_gate()
        cgate3(np.pi / 2) | cir([2, 4, 6])

        matrix = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.complex128)
        cgate4 = Unitary(matrix)
        cgate4.build_gate()
        cgate4 | cir
        assert len(cir.gates) == 19

        # append composite gate
        qft_gate = QFT(3)
        qft_gate | cir([0, 1, 3])

        cgate = CompositeGate()
        CCRz(1) | cgate([0, 1, 2])
        assert len(cir.gates) == 22

        # append gate by qubits/qureg
        S | cir(qureg[0])
        CX | cir(qureg[1, 3])
        assert len(cir.gates) == 24

        # append cir to cir
        cir1 = Circuit(5)
        H | cir1
        cir2 = Circuit(3)
        cir1 | cir2
        assert cir2.width() == 8 and cir2.size() == 5

        cir1 = Circuit(3)
        H | cir1
        cir2 = Circuit(3)
        cir1 | cir2
        assert cir1.width() == 3 and cir2.depth() > cir1.depth()

        cir1 = Circuit(5)
        H | cir1
        cir2 = Circuit(2)
        cir1([2, 3]) | cir2
        assert cir1.width() == 5 and cir2.gates[-1].targs == [3]

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
        assert cir.size() == 60
        assert sub_cir_without_remove.width() == 3

        sub_cir_with_remove = cir.sub_circuit(max_size=120, qubit_limit=[0, 3])
        assert cir.size() + sub_cir_with_remove.size() == 80
        assert sub_cir_without_remove.width() == 3

        # transform circuit to composite gate
        composite_gate = cir.to_compositegate()
        cir_new = TestCircuit.based_circuit
        cir_new.extend(composite_gate)
        assert cir.size() == cir_new.size()

        # append trigger circuit

        # add qubits to circuit with ancillary_qubit or not
        cir.add_qubit(5, is_ancillary_qubit=False)
        assert cir.width() == 10
        cir.add_qubit(5, is_ancillary_qubit=True)
        assert cir.width() == 15
        assert cir.ancilla_qubits == [10, 11, 12, 13, 14]

        # reset qubits to circuit
        cir.reset_qubits
        assert cir.width() == 15
        
        # extend gate/circuit to circuit
        cir1 = Circuit(1)
        cgate = CompositeGate(GateType.h)
        H | cgate(1)
        cir1.extend(cgate)
        assert cir1.size() == 1

        cir1.extend(cir1)
        assert cir1.size() == 2

        # append gate to circuit
        cir2 = Circuit(3)
        cir2.append(H)
        assert cir2.size() == 3

        # insert gate to circuit
        c_insert = CompositeGate()
        H | c_insert(0)
        cir2.insert(c_insert, 1)
        c_insert | cir2
        
        c_insert2 = CX & [1, 2]
        cir2.insert(c_insert2, 2)
        assert cir2.size() == 6
        
    def test_circuit_info(self):
        # add all type of gates to circuit
        cir = Circuit(10)
        H | cir(0)
        Hy | cir(0)
        S | cir(1)
        S_dagger | cir(1)
        X | cir(2)
        Y | cir(2)
        Z | cir(2)
        SX | cir(3)
        SY | cir(3)
        SW | cir(3)
        ID | cir(4)
        U1(np.pi / 2) | cir(5)
        U2(np.pi / 2, 0) | cir(5)
        U3(np.pi / 2, 0, 0) | cir(5)
        Rx(np.pi / 2) | cir(6)
        Ry(np.pi / 2) | cir(6)
        Rz(np.pi / 2) | cir(6)
        T | cir(7)
        T_dagger | cir(7)
        Phase(np.pi / 2) | cir(8)
        GPhase(np.pi / 2) | cir(8)
        CZ | cir([0, 1])
        CX | cir([1, 2])
        CH | cir([2, 3])
        CRz(np.pi / 2) | cir([0, 1])
        CU1(np.pi / 2) | cir([1, 2])
        CU3(np.pi / 2, 0, 0) | cir([3, 4])
        FSim(np.pi / 2, 0) | cir([3, 4])
        Rxx(np.pi / 2) | cir([5, 6])
        Ryy(np.pi / 2) | cir([6, 7])
        Rzz(np.pi / 2) | cir([7, 8])
        Measure | cir(7)
        Reset | cir(8)
        Barrier | cir(8)
        Swap | cir([0, 1])
        iSwap | cir([0, 1])
        iSwap_dagger | cir([0, 1])
        sqiSwap | cir([0, 1])
        CCX | cir([0, 1, 2])
        CCZ | cir([3, 4, 5])
        CCRz(np.pi / 2) | cir([6, 7, 8])
        CSwap | cir([7, 8, 9])
        
        cir.set_precision("single")
        assert cir.width() == 10
        assert cir.size() == 42
        assert cir.depth() == 11
        assert len(cir.qubits) == 10
        assert cir.count_1qubit_gate() == 24
        assert cir.count_2qubit_gate() == 14
        assert cir.count_gate_by_gatetype(GateType.measure) == 1
        # assert len(cir.gates) == cir.size()
        # assert len(cir.fast_gates) == cir.size()
        assert len(cir.flatten_gates()) == cir.size()
        assert cir.precision == "single"
        # for gate in cir.gates:
        #     gate.precision == "single"

    def test_noise_circuit(self):
        cir = TestCircuit.based_circuit
        H | cir
        Z | cir(2)
        CX | cir([0, 1])
        pauil_error_rate = 0.6
        nm = NoiseModel()
        bf_err = BitflipError(pauil_error_rate)
        pf_err = PhaseflipError(pauil_error_rate)
        single_readout = ReadoutError(np.array([[0.8, 0.2], [0.2, 0.8]]))
        bits_err = PauliError(
            [('zy', pauil_error_rate), ('xi', 1 - pauil_error_rate)],
            num_qubits=2
        )
        nm.add_noise_for_all_qubits(bf_err, ['h'])
        nm.add(pf_err, ['z'], [0, 1, 3])
        nm.add_readout_error(single_readout, [1, 2])
        nm.add(bits_err, ['cx'], [0, 1])

        noised_circuit = nm.transpile(cir, accumulated_mode=True)
        assert type(noised_circuit.gates) == NoiseGate
        assert noised_circuit.size() == len(noised_circuit.gates)
        
        noised_circuit = nm.transpile(cir)
        assert type(noised_circuit.gates) == NoiseGate
        assert noised_circuit.size() == cir.size() + len(noised_circuit.gates)

    def test_circuit_matrix_product(self):  # move to gate ut
        cir = Circuit(5)
        mp_gate = CZ & [1, 3]
        mp_data = mp_gate.expand(cir.width())

        assert mp_data.shape == (1 << 5, 1 << 5)

    def test_circuit_decomposition(self):
        build_gate_typelist = [GateType.ccrz, GateType.cswap, GateType.ccz, GateType.ccx]
        cir = Circuit(TestCircuit.qubits)
        cir.random_append(30)
        cir.random_append(10, build_gate_typelist)
        QFT(5) | cir
        cir.gate_decomposition()
        for gate in cir.gates:
            assert gate.controls + gate.targets < 3

    def test_circuit_matrix(self): # add set_precision
        from QuICT.simulation.state_vector import StateVectorSimulator
        from QuICT.simulation.unitary import UnitarySimulator

        cir = Circuit(TestCircuit.qubits)
        cir.random_append(10)
        state_vector_cir = StateVectorSimulator().run(cir)

        cir_matrix = cir.matrix()
        state_vector_matrix = UnitarySimulator().run(cir_matrix)

        assert np.allclose(state_vector_cir, state_vector_matrix)


if __name__ == "__main__":
    unittest.main()
