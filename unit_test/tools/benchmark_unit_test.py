import os
import unittest

from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.core.virtual_machine import InstructionSet
from QuICT.simulation.state_vector import StateVectorSimulator


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("The QuICT benchmark unit test start!")

    @classmethod
    def tearDownClass(cls) -> None:
        print("The QuICT benchmark unit test finished!")

    def test_validate_circuits(self):
        benchmark = QuICTBenchmark()
        circuits_list = benchmark.get_circuits(quantum_machine_info={"qubits_number": 5})
        amp_results_list = []
        for circuit in circuits_list:
            simulator = StateVectorSimulator()
            sim_results = simulator.run(circuit)
            amp_results_list.append(sim_results)

        entropy_QV_score = benchmark._entropy_QV_score(circuits_list, amp_results_list)
        valid_circuits_list = benchmark._filter_system(entropy_QV_score)

        assert len(circuits_list) == len(valid_circuits_list)
        assert circuits_list[5].name == valid_circuits_list[5]

    def test_circuits_number(self):
        layout = Layout.load_file(os.path.dirname(os.path.abspath(__file__)) + "/../../example/layout/grid_3x3.json")
        Inset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])

        benchmark = QuICTBenchmark()
        circuits_list = benchmark.get_circuits(
            quantum_machine_info={"qubits_number": 2, "layout_file": layout, "Instruction_Set": Inset},
            level=1,
            mapping=True,
            gate_transform=True
        )
        assert len(circuits_list) == 84


if __name__ == "__main__":
    unittest.main()
