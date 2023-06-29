import random
import unittest

from QuICT.core import Qureg


class TestQubit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("The Qubit unit test start!")

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Qubit unit test finished!")

    def test_qubit_attr(self):
        # unqiue test
        qureg = Qureg(10)
        id_list = [qubit.id for qubit in qureg]
        assert len(set(id_list)) == len(id_list)

        qureg.set_fidelity([0.99] * 10)
        qureg.set_gate_fidelity(0.987)
        qureg.set_gate_duration(20.31)
        qureg.set_t2_time([5.48] * 10)

        measure_result = 0
        for qubit in qureg:
            measure = random.random() > 0.5
            measure_result <<= 1
            if measure:
                measure_result += 1
                qubit.measured = 1
            else:
                qubit.measured = 0

        assert int(qureg) == measure_result

    def test_qubit_call(self):
        qureg = Qureg(10)
        idx = random.sample(range(10), 3)
        squbit_ids = [qureg[i].id for i in idx]
        cqureg = qureg(idx)
        for cq in cqureg:
            assert cq.id in squbit_ids

    def test_qubit_idx(self):
        qureg = Qureg(5)
        qbit = qureg[3]

        assert qureg.index(qbit) == 3

        sqreg = qureg[0, 3]
        assert qureg.index(sqreg) == [0, 3]

    def test_qureg_operation(self):
        q1 = Qureg(5)
        q2 = Qureg(5)
        q_add = q1 + q2
        assert len(q_add) == (len(q1) + len(q2))

        assert q1 == q1
        assert not q1 == q2


if __name__ == "__main__":
    unittest.main()
