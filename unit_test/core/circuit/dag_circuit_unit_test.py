import unittest

from QuICT.core import Circuit
from QuICT.core.gate import *

class TestDagCircuit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Dag Circuit unit test start!")
        cls.circuit = Circuit(5)
        cls.circuit.random_append(20)

    @classmethod
    def tearDownClass(cls) -> None:
        print("Dag Circuit unit test finished!")

    def test_dag_circuit(self):
            dag_cir = TestDagCircuit.circuit.get_DAG_circuit()
            dag_cir.size == TestDagCircuit.circuit.size()
            edge_list = dag_cir.edges()
            gs = TestDagCircuit.circuit.gates

            for start, end in edge_list:
                assert not gs[start].commutative(gs[end])
                forward = True
            for f in range(start + 1, end, 1):
                if not gs[start].commutative(gs[f]):
                    forward = False
                    break

            if not forward:
                for b in range(end - 1, start, -1):
                    if not gs[end].commutative(gs[b]):
                        assert 0
            assert 1

if __name__ ==" __main__":
    unittest.main()
