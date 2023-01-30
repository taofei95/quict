import os
import unittest

from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("The QuICT benchmark unit test start!")

    @classmethod
    def tearDownClass(cls) -> None:
        print("The QuICT benchmark unit test finished!")

    def test_output_file_type(self):
        def sim(cir):
            result = ConstantStateVectorSimulator().run(cir).get()
            return result

        QuICTBenchmark(device="GPU", output_file_type="txt").run(
            simulator_interface=sim,
            quantum_machine_info={"qubits_number": 3}
        )
        file_name_txt = os.listdir("./benchmark")

        QuICTBenchmark(device="GPU", output_file_type="excel").run(
            simulator_interface=sim,
            quantum_machine_info={"qubits_number": 3}
        )
        file_name_excel = os.listdir("./benchmark")

        assert file_name_txt != file_name_excel

    def test_qcda_choice(self):
        layout = Layout.load_file(os.path.dirname(os.path.abspath(__file__)) + "/../../example/layout/grid_3x3.json")
        Inset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])

        # no mapping and no gate transform
        QuICTBenchmark().get_circuits(quantum_machine_info={"qubits_number": 2})
        assert True

        # mapping and no gate transform
        QuICTBenchmark().get_circuits(
            quantum_machine_info={"qubits_number": 2, "layout_file": layout},
            mapping=True
        )
        assert True

        # no mapping and gate transform
        QuICTBenchmark().get_circuits(
            quantum_machine_info={"qubits_number": 2, "Instruction_Set": Inset},
            gate_transform=True
        )
        assert True

        # mapping and gate transform
        QuICTBenchmark().get_circuits(
            quantum_machine_info={"qubits_number": 2, "layout_file": layout, "Instruction_Set": Inset},
            mapping=True,
            gate_transform=True
        )
        assert True


if __name__ == "__main__":
    unittest.main()
