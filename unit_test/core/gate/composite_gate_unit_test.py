#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/18 12:03 上午
# @Author  : Li Kaiqi
# @File    : composite_gate_unit_test
import unittest

from QuICT.core.gate import *


class TestCompositeGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Composite Gate unit test start!")
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
        cls.default_parameter_gates_for_call_test = [U1, U2, CU3, FSim, CCRz]


    @classmethod
    def tearDownClass(cls) -> None:
        print("The Composite Gate unit test finished!")

    def test_compositegate_build(self):
        qubit_range = list(range(10))
        for _ in range(10):
            cgate = CompositeGate()
            size, depth = 0, [0] * 10
            count_gate1, count_gate2 = 0, 0
            for _ in range(100):
                # choice Gate[] and random index
                gate_args_choice = np.random.randint(0, 3)
                if gate_args_choice == 0:
                    target_gate_list = self.default_1_qubits_gate
                    target_gate_indexes = [np.random.choice(qubit_range)]
                    count_gate1 += 1
                    depth[target_gate_indexes[0]] += 1
                elif gate_args_choice == 1:
                    target_gate_list = self.default_2_qubits_gate
                    target_gate_indexes = list(np.random.choice(qubit_range, 2, False))
                    count_gate2 += 1
                    new_depth = max(depth[target_gate_indexes[0]], depth[target_gate_indexes[1]]) + 1
                    depth[target_gate_indexes[0]] = new_depth
                    depth[target_gate_indexes[1]] = new_depth
                else:
                    target_gate_list = self.default_3_qubits_gate
                    target_gate_indexes = list(np.random.choice(qubit_range, 3, False))
                    new_depth = max(depth[target_gate_indexes[0]], depth[target_gate_indexes[1]], depth[target_gate_indexes[2]]) + 1
                    depth[target_gate_indexes[0]] = new_depth
                    depth[target_gate_indexes[1]] = new_depth
                    depth[target_gate_indexes[2]] = new_depth

                gate_type_choice = np.random.randint(0, len(target_gate_list))
                target_gate = gate_builder(target_gate_list[gate_type_choice], random_params=True)

                # different build way
                # using original gate with call index/assigned index
                rand_i = np.random.randint(0, 3)
                if rand_i == 0:
                    target_gate | cgate(target_gate_indexes)
                elif rand_i == 1:
                    target_gate & target_gate_indexes | cgate
                else:
                    cgate.append(target_gate & target_gate_indexes)

                size += 1

                # parameter gate
                is_add_param_call = np.random.rand()
                if is_add_param_call > 0.5:
                    target_pgate = self.default_parameter_gates_for_call_test[np.random.randint(0, len(self.default_parameter_gates_for_call_test))]
                    random_params = [np.random.rand() for _ in range(target_pgate.params)]
                    target_pgate_indexes = list(np.random.choice(qubit_range, target_pgate.controls + target_pgate.targets, False))
                    target_pgate(*random_params) | cgate(target_pgate_indexes)

                    size += 1
                    if len(target_pgate_indexes) == 1:
                        count_gate1 += 1
                        depth[target_pgate_indexes[0]] += 1
                    elif len(target_pgate_indexes) == 2:
                        count_gate2 += 1
                        new_depth = max(depth[target_pgate_indexes[0]], depth[target_pgate_indexes[1]]) + 1
                        depth[target_pgate_indexes[0]] = new_depth
                        depth[target_pgate_indexes[1]] = new_depth
                    else:
                        new_depth = max(depth[target_pgate_indexes[0]], depth[target_pgate_indexes[1]], depth[target_pgate_indexes[2]]) + 1
                        depth[target_pgate_indexes[0]] = new_depth
                        depth[target_pgate_indexes[1]] = new_depth
                        depth[target_pgate_indexes[2]] = new_depth

            assert cgate.size() == size
            assert cgate.depth() == np.max(depth)
            assert cgate.count_1qubit_gate() == count_gate1
            assert cgate.count_2qubit_gate() == count_gate2

            qft = QFT(5)
            qft.count_gate_by_gatetype(GateType.h) == 5

            iqft = IQFT(5)
            iqft.count_gate_by_gatetype(GateType.h) == 5
            assert qft.size() + iqft.size() == 30

    def test_compositegate_matrix(self):
        test_gate = CompositeGate()
        H | test_gate(0)
        H | test_gate(1)
        H | test_gate(2)
        CX | test_gate([0, 1])
        CX | test_gate([0, 2])

        matrix = test_gate.matrix()
        assert matrix.shape == (1 << 3, 1 << 3)

    def test_compositegate_merge(self):
        # composite gate | composite gate
        cgate1 = CompositeGate()
        X | cgate1(0)
        Y | cgate1(1)
        Z | cgate1(2)
        cgate2 = CompositeGate()
        CX | cgate2([0, 1])
        CY | cgate2([1, 2])
        CZ | cgate2([2, 3])
        cgate1 | cgate2
        assert cgate2.size() == 6

        # composite gate ^ composite gate
        cgate1 ^ cgate2
        assert cgate2.size() == 9
        cgate2 ^ cgate2
        assert cgate2.size() == 18

        with cgate2:
            SX & 0
            SY & 1
            SW & 2
        assert cgate2.size() == 21

    def test_circuit_modify(self):
        cgate = CompositeGate(5)

        # insert gate to circuit
        cgate.insert(H & 0, 0)
        cgate.insert(cgate, 1)
        cgate.insert(CU3(1, 0, 1) & [1, 2], 2)
        assert cgate.fast_gates[0][0].type == GateType.h

        # pop gate from Circuit
        gate = cgate.pop(-1)
        assert cgate.size() == 2 and gate.type == GateType.cu3

        # Adjust gate
        cgate.adjust(-1, [3])
        cgate.adjust(-1, [1], True)

        # extend gate/circuit
        cgate = CompositeGate()
        CX | cgate([1, 0])
        cgate1 = CompositeGate()
        CH | cgate1([0, 1])
        cgate.extend(cgate1)
        assert cgate.width() == 2 and cgate.size() == 2

        # inverse cgate
        cgate_inverse = cgate.inverse()
        cgate_inverse | cgate(list(range(5)))
        assert np.allclose(cgate.matrix(), np.identity(1 << 2, dtype=np.complex128))

if __name__ == "__main__":
    unittest.main()
