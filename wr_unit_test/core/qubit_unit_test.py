import random
import unittest

from QuICT.core import Circuit, Qureg
from QuICT.core.gate import BasicGate


class TestQubit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("circuit_unit_test begin")


    @classmethod
    def tearDownClass(cls) -> None:
        print("circuit_unit_test finish")


    def test_qubit_attr(self):
        qureg = Qureg(10)
        id_list = [qubit.id for qubit in qureg]
        assert len(set(id_list))==len(id_list)

        for qubit in qureg:
            measure = random.random()>0.5
            if measure:
                qubit.measured =1
            else:
                qubit.measured=0
            assert int(qubit) == qubit.measured
            assert bool(qubit) == measure
    def test_qubit_call(self):
        qureg = Qureg(10)
        idx = random.sample(range(10),3)
        squbit_ids = [qureg[i].id for i in idx]
        cqureg = qureg(idx)
        for cq in cqureg:
            assert cq.id in squbit_ids
    def test_qureg_int(self):
        qureg = Qureg(5)
        for qubit in qureg:
            qubit.measured = 1
        assert int(qureg) == 31
    def test_qureg_operation(self):
        q1=Qureg(5)
        q2=Qureg(5)
        q_add = q1+q2
        assert len(q_add) == (len(q1)+len(q2))

        assert q1 == q1
        assert not q1 == q2

        diff_q = q1.diff(q2)
        assert diff_q == q2



if __name__ == "__main":
    unittest.main()