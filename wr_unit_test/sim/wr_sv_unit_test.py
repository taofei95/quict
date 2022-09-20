import unittest
import os

from QuICT.core import Circuit
from QuICT.core.gate import *
from quict.QuICT.simulation import state_vector
from quict.QuICT.simulation.state_vector.cpu_simulator import cpu
from quict.QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator
from quict.QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
from quict.QuICT.simulation.utils.result import Result



# @unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestSVSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls): 
        print('sv simulator begin')
    if os.environ.get("test_with_gpu"):
        from QuICT.simulation.state_vector import ConstantStateVectorSimulator   

    @classmethod
    def tearDownClass(cls): 
        print('sv simulator finished')

    def test_constant_statevectorsimulator(self):
        qubit_num = 5

        circuit = Circuit(qubit_num)
        QFT.build_gate(qubit_num) | circuit
        QFT.build_gate(qubit_num) | circuit
        QFT.build_gate(qubit_num) | circuit

        simulator = ConstantStateVectorSimulator(
            precision="double",
            gpu_device_id=0,
            sync=True
        )
        _ = simulator.run(circuit)
        assert 1

        for _ in range(30):
            simulator = CircuitSimulator()
            _ = simulator.run(circuit)
            measure_res = simulator.sample(circuit)
            print(measure_res)
            assert 1 
        
if __name__ == "__main__":
    unittest.main()
