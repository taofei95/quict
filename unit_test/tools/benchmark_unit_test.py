import random
import unittest

from QuICT.benchmark.benchmark import QuantumMachinebenchmark
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.core.virtual_machine import InstructionSet
from QuICT.core.virtual_machine.virtual_machine import VirtualQuantumMachine
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("The quantum machine benchmark unit test start!")

    @classmethod
    def tearDownClass(cls) -> None:
        print("The quantum machine benchmark unit test finished!")

    def test_circuit_number(self):
        iset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])
        layout = Layout.linear_layout(5)
        vqm = VirtualQuantumMachine(
            qubits=5,
            instruction_set=iset,
            layout=layout,
        )
        benchmark = QuantumMachinebenchmark()
        # level1, no qcda for algorithm circuit
        circuits_list = benchmark.get_circuits(quantum_machine_info=vqm)
        assert len(circuits_list) == 20  # random4 + benchmark16
        # level3, and qcda for algorithm circuit
        circuits_list = benchmark.get_circuits(quantum_machine_info=vqm, level=3, enable_qcda_for_alg_cir=True)
        assert len(circuits_list) == 35  # random4 + benchmark16 + alg17
        circuits_list = benchmark.get_circuits(quantum_machine_info=vqm, is_measure=True)
        random_test_cir = random.choice(circuits_list)
        assert random_test_cir.circuit.gates[-1].type == GateType.measure

    def test_benchlib(self):
        def sim_interface(cir):
            sim = StateVectorSimulator()
            sim.run(cir)
            amp_list = sim.sample(1)
            return amp_list

        iset = InstructionSet(GateType.cx, [GateType.h])
        layout = Layout.linear_layout(5)
        vqm = VirtualQuantumMachine(
            qubits=5,
            instruction_set=iset,
            layout=layout,
        )
        benchmark = QuantumMachinebenchmark()
        circuits_list = benchmark.get_circuits(quantum_machine_info=vqm)
        assert len(circuits_list) == 20
        for cir in circuits_list:
            circuit = cir.circuit
            if cir.type != "benchmark":
                circuit_name1 = f"{cir.type}+{cir.field}+w{cir.width}_s{cir.size}_d{cir.depth}+level{cir.level}"
                assert circuit.name == circuit_name1
            else:
                circuit_name2 = f"{cir.type}+{cir.field}+w{cir.width}_s{cir.size}_d{cir.depth}_v{cir.bench_cir_value}+\
                    level{cir.level}"
                assert circuit_name2 != circuit_name1
            cir.machine_amp = sim_interface(cir.circuit)
            assert cir.fidelity == cir.machine_amp[0]
            assert cir.benchmark_score <= cir.qv * cir.fidelity


if __name__ == "__main__":
    unittest.main()
