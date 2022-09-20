import unittest

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import *
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulation

class TestGPUSimulator(unittest.TestCase): 
    @classmethod
    def setUpClass(cls): 
        print('simulator sample test begin!')

    @classmethod
    def tearDownClass(cls): 
        print('simulator sample test finished!')

    def SampleTest():
        qubit_num = 4
        cir = Circuit(qubit_num)
        H | cir(0)
        for i in range(qubit_num - 1):
            CX | cir([i, i+1])

        simulator = DensityMatrixSimulation()
        _ = simulator.run(cir)
        a = simulator.sample(100)
        print(a)
        assert a[0] + a[-1] == 100

        
if __name__=="__main__":
  unittest.main()
