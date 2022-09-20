import unittest
import numpy as np
from QuICT.algorithm import SyntheticalUnitary
from QuICT.core.gate import *
from QuICT.simulation.unitary import UnitarySimulator
from QuICT.core.circuit.circuit import Circuit
from quict.QuICT.simulation import unitary
from quict.QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator
from quict.QuICT.simulation.utils.result import Result

class Testunitarysim(unittest.TestCase):
    @classmethod
    def setUpClass(cls): 
        print('unitary simulator unit test begin')

    @classmethod
    def tearDownClass(cls): 
        print('unitary simulator unit test finished')
 
    def test_unitary_generate(self):
        qubit = 5
        gate_number = 100
        circuit = Circuit(qubit)
        circuit.random_append(gate_number)

        circuit_unitary = SyntheticalUnitary.run(circuit)
        sim = UnitarySimulator()
        result_mat = sim.get_unitary_matrix(circuit)
        assert np.allclose(circuit_unitary, result_mat)

        _ = sim.run(circuit)
        assert 1
        for _ in range(30):
            simulator = CircuitSimulator()
            _ = simulator.run(circuit)
            measure_res = simulator.sample(circuit)
            print(measure_res)
            assert 1 
        
    print(unittest.TestCase)
    
if __name__ == "__main__":
    unittest.main()
