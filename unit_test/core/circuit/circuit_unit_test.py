import unittest
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


class TestCircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Circuit unit test start!")
        cls.default_1_qubits_gate = [GateType.x, GateType.y, GateType.z]
        cls.default_2_qubits_gate = [GateType.cx, GateType.cz]
        cls.default_parameter_gates_for_call_test = [U1, U2]

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Circuit unit test finished!")

    def test_circuit_build(self):
        qubit_range = list(range(10))
        for _ in range(10):
            cir = Circuit(10)
            size, depth = 0, [0] * 10
            count_gate1, count_gate2 = 0, 0
            for _ in range(100):
                # choice Gate[] and random index, TODO: add 3-qubits gate here
                gate_args_choice = np.random.randint(0, 2)
                if gate_args_choice == 0:
                    target_gate_list = self.default_1_qubits_gate
                    target_gate_indexes = [np.random.choice(qubit_range)]
                    count_gate1 += 1
                    depth[target_gate_indexes[0]] += 1
                else:
                    target_gate_list = self.default_2_qubits_gate
                    target_gate_indexes = list(np.random.choice(qubit_range, 2, False))
                    count_gate2 += 1
                    new_depth = max(depth[target_gate_indexes[0]], depth[target_gate_indexes[1]]) + 1
                    depth[target_gate_indexes[0]] = new_depth
                    depth[target_gate_indexes[1]] = new_depth

                gate_type_choice = np.random.randint(0, len(target_gate_list))
                target_gate = gate_builder(target_gate_list[gate_type_choice], random_params=True)

                # different build way
                # using original gate with call index/assigned index
                rand_i = np.random.randint(0, 3)
                if rand_i == 0:
                    target_gate | cir(target_gate_indexes)
                elif rand_i == 1:
                    target_gate & target_gate_indexes | cir
                else:
                    cir.append(target_gate & target_gate_indexes)

                size += 1

                # parameter gate
                is_add_param_call = np.random.rand()
                if is_add_param_call > 0.5:
                    target_pgate = self.default_parameter_gates_for_call_test[np.random.randint(0, len(self.default_parameter_gates_for_call_test))]
                    random_params = [np.random.rand() for _ in range(target_pgate.params)]
                    target_pgate_indexes = list(np.random.choice(qubit_range, target_pgate.controls + target_pgate.targets, False))
                    target_pgate(*random_params) | cir(target_pgate_indexes)

                    size += 1
                    if len(target_pgate_indexes) == 1:
                        count_gate1 += 1
                        depth[target_pgate_indexes[0]] += 1
                    else:
                        count_gate2 += 1
                        new_depth = max(depth[target_pgate_indexes[0]], depth[target_pgate_indexes[1]]) + 1
                        depth[target_pgate_indexes[0]] = new_depth
                        depth[target_pgate_indexes[1]] = new_depth

            assert cir.depth() == np.max(depth)
            # CompositeGate
            qft_gate = QFT(3)
            qft_gate | cir([0, 1, 2])
            size += qft_gate.size()

            # Circuit
            cir_new = Circuit(3)
            qft_gate | cir_new
            cir_new | cir([9, 8, 7])
            size += qft_gate.size()

            # Random/Supremacy append
            cir.random_append()
            size += 10

            cir.supremacy_append()
            size += 124

            assert cir.size() == size

    # def test_sub_circuit(self):
    #     # append sub circuit
    #     cir = Circuit(5)
    #     cir.random_append(100)

    #     # qubits limit
    #     sub_cir_qubit = cir.sub_circuit(qubit_limit=[0, 1, 2])
    #     assert cir.size() == 100 and sub_cir_qubit.width() == 3

    #     # size limit
    #     sub_cir_size = cir.sub_circuit(max_size=20)
    #     assert sub_cir_size.size() == 20

    #     # gate limit
    #     sub_cir_gate = cir.sub_circuit(gate_limit=[GateType.cx])
    #     assert sub_cir_gate.size() == cir.count_gate_by_gatetype(GateType.cx)

    # def test_to_compositegate(self):
        # transform circuit to composite gate
        cir_trans = Circuit(3)
        cir_trans.random_append(10)
        composite_gate = cir_trans.to_compositegate()

        assert np.allclose(cir_trans.matrix(), composite_gate.matrix())

    def test_circuit_modify(self):
        cir = Circuit(5)
        cir.random_append(10)

        # add qubits to circuit with ancillary_qubit or not
        cir.add_qubit(5, is_ancillary_qubit=False)
        assert cir.width() == 10
        cir.add_qubit(5, is_ancillary_qubit=True)
        assert cir.width() == 15 and cir.ancilla_qubits == list(range(10, 15))

        # reset qubits to circuit
        cir.reset_qubits

        # insert gate to circuit
        cir.insert(H & 0, 0)
        cir.insert(H & 0, 11)
        cir.insert(CU3(1, 0, 1) & [1, 2], 12)
        assert cir.fast_gates[0][0].type == GateType.h

        # pop gate from Circuit
        gate = cir.pop(-1)
        assert cir.size() == 12 and gate.type == GateType.cu3

        # Adjust gate
        cir.adjust(-1, [3])
        cir.adjust(-1, [1], True)
        assert cir.gates[-1].targ == 4

    def test_inverse_circuit(self):
        cir = Circuit(3)
        cir.random_append(10)

        # inverse circuit
        cir_inverse = cir.inverse()
        cir_inverse | cir([0, 1, 2])
        assert np.allclose(cir.matrix(), np.identity(1 << 3, dtype=np.complex128))

    def test_circuit_decomposition(self):
        build_gate_typelist = [GateType.ccrz, GateType.cswap, GateType.ccz, GateType.ccx]
        cir = Circuit(5)
        cir.random_append(10, build_gate_typelist)
        QFT(5) | cir
        cir.gate_decomposition()
        for gate in cir.gates:
            assert isinstance(gate, BasicGate) and gate.controls + gate.targets < 3

    def test_circuit_matrix(self):
        from QuICT.simulation.state_vector import StateVectorSimulator
        from QuICT.simulation.unitary import UnitarySimulator

        cir = Circuit(5)
        cir.random_append(10)
        state_vector_cir = StateVectorSimulator().run(cir)

        cir_matrix = cir.matrix()
        state_vector_matrix = UnitarySimulator().run(cir_matrix)
        assert np.allclose(state_vector_cir, state_vector_matrix)

    def test_circuit_qasm(self):
        cir = Circuit(5)
        cir.random_append(20)

        qasm_str = cir.qasm()

        qasm_cir = OPENQASMInterface.load_string(qasm_str).circuit
        assert qasm_cir.size() == 20 and qasm_cir.width() == 5


if __name__ == "__main__":
    unittest.main()
