import os
import numpy as np

from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator


def sim(cir):
    result = ConstantStateVectorSimulator().run(cir).get()
    return result


def benchmark():
    layout = Layout.load_file(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/layout/grid_3x3.json")
    Inset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])

    benchmark = QuICTBenchmark(device="GPU", output_path="./example/benchmark_output")
    benchmark.run(
        simulator_interface=sim,
        quantum_machine_info={"qubits_number": 3, "layout_file": layout, "Ins_Set": Inset},
        level=3,
        mapping=True,
        gate_transform=True
    )
    print(os.listdir("./example/benchmark_output"))

    machine_simulation_results = []
    path = "example/benchmark/simulation_data/"
    for result in os.listdir(path):
        machine_simulation_results.append(np.load(path + result))

    benchmark = QuICTBenchmark(output_path="./example/benchmark_output")
    cirs = benchmark.get_circuits(quantum_machine_info={"qubits_number": 2, "layout_file": [], "Ins_Set": []}, level=1)
    benchmark.evaluate(circuits_list=cirs, amp_results_list=machine_simulation_results)
    print(os.listdir("./example/benchmark_output"))


if __name__ == "__main__":
    benchmark()
