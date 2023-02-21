import os
import shutil
import uuid
import unittest

from QuICT.tools.circuit_library import CircuitLib


class TestCircuitLibrary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('Circuit Library unit test begin!')

    @classmethod
    def tearDownClass(cls):
        print('Circuit Library unit test finished!')

    def test_circuit(self):
        # Test Circuit Library with output circuit
        cir_lib = CircuitLib(output_type="circuit")

        # Test interface get_circuit
        cirs = cir_lib.get_circuit("algorithm", "qft", 10)
        for cir in cirs:
            name = cir.name
            type_, classify, _ = name.split('+')
            assert type_ == "algorithm" and classify == "qft", "error circuit get from get_circuit."

        # Test get_algorithm_circuit
        cirs = cir_lib.get_algorithm_circuit("grover", [3, 5, 7], max_size=100, max_depth=20)
        for cir in cirs:
            assert cir.width() in [3, 5, 7], "Error width get from get_algorithm_circuit."
            assert cir.size() <= 100 and cir.depth() <= 20, "Error size and depth get from get_algorithm_circuit."

        # Test get_random_circuit
        cirs = cir_lib.get_random_circuit("diag", [3, 5, 7], max_size=40, max_depth=20)
        for cir in cirs:
            assert cir.width() in [3, 5, 7], "Error width get from get_algorithm_circuit."
            assert cir.size() <= 40 and cir.depth() <= 20, "Error size and depth get from get_random_circuit."

        # Test get_benchmark_circuit
        cirs = cir_lib.get_benchmark_circuit("highly_entangled", [3, 5], max_size=20, max_depth=15)
        for cir in cirs:
            assert cir.width() in [3, 5], "Error width get from get_algorithm_circuit."
            assert cir.size() <= 20 and cir.depth() <= 15, "Error size and depth get from get_benchmark_circuit."

        # Test get_template_circuit
        cirs = cir_lib.get_template_circuit(qubits_interval=5, max_size=20, max_depth=15)
        for cir in cirs:
            assert cir.width() <= 5, "Error width get from get_algorithm_circuit."
            assert cir.size() <= 20 and cir.depth() <= 15, "Error size and depth get from get_template_circuit."

    def test_qasm(self):
        # Test Circuit Library with output circuit
        cir_lib = CircuitLib(output_type="qasm")

        qasms = cir_lib.get_circuit("algorithm", "maxcut", 10)
        for qasm in qasms:
            assert isinstance(qasm, str)

    def test_output(self):
        output_path_id = str(uuid.uuid4())
        output_path = f"./temp_list_{output_path_id}"
        cir_lib = CircuitLib(output_type="file", output_path=output_path)
        cir_lib.get_circuit("template", "template", 3, 6, 5)

        files_name = os.listdir(output_path)
        for fname in files_name:
            classify, width, size, depth, _ = fname.split("_")
            assert classify == "template"
            width = int(width[1])
            size = int(size[1])
            depth = int(depth[1])
            assert width <= 3 and size <= 6 and depth <= 5

        shutil.rmtree(output_path)


if __name__ == "__main__":
    unittest.main()
