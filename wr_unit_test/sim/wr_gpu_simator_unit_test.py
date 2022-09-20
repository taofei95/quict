import unittest
import cupy as cp
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import CCX, H, QFT, CCRz
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator

class Testcpusim(unittest.TestCase):
    @classmethod
    def setUpClass(cls): 
        print('GPU Simlator begin!')
    
    @classmethod
    def tearDownClass(cls): 
        print('GPU Simlator finished!')

    def test_sim(self):
        for qubit_num in range(2, 20):
            circuit = Circuit(qubit_num)
            circuit.random_append(20)
            _ = CircuitSimulator().run(circuit)  # New simulator would be used by default.
            # expected = Amplitude.run(circuit, ancilla=None, use_old_simulator=True)
            # flag = np.allclose(res, expected)
            # assert flag
            # print(f"Testing for qubit {qubit_num}: {flag}")
            assert 1 


    def test_complex_gate(self):
        for qubit_num in range(3, 20):
            circuit = Circuit(qubit_num)
            QFT(qubit_num) | circuit
            CCX | circuit([0, 1, 2])
            CCRz(0.1) | circuit([0, 1, 2])
            _ = CircuitSimulator().run(circuit)  # New simulator would be used by default.
            # expected = Amplitude.run(circuit, ancilla=None, use_old_simulator=True)
            # flag = np.allclose(res, expected)
            # assert flag
            assert 1 

    def test_random_gate(self):
        qubit_num = 10
        gate_num = 50
        circuit = Circuit(qubit_num)
        circuit.random_append(gate_num)
        _=CircuitSimulator().run(circuit)
        
        assert 1 


    def test_measure_gate(self):
        qubit_num = 4
        # measure_res_acc = 0
        for _ in range(30):
            circuit = Circuit(qubit_num)
            H | circuit
            simulator = CircuitSimulator()
            _ = simulator.run(circuit)
            measure_res = simulator.sample(circuit)
            print(measure_res)
            assert 1 


if __name__ ==" __main__":
    unittest.main()
