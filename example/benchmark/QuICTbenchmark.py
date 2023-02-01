import os

from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator

layout = Layout.load_file(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/layout/grid_3x3.json")
Inset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])


def simulation(circuit):
    """Simulate the quantum physical machine interface.

    Args:
        circuit: A quantum circuit.

    Returns:
        List: Amplitude results or Sample results for circuits.
    """
    simulator = CircuitSimulator()
    sim_results = simulator.run(circuit)
    # sim_sample = simulator.sample(circuit)
    return sim_results


def Step_benchmark():
    # First method: Initialize -> Get the circuits -> Results Analysis.

    # Step1: initialize QuICTBenchmark.
    benchmark = QuICTBenchmark(
        device="CPU",                 # choose simulation device, one of [CPU, GPU]
        output_path="./benchmark",    # The path of the Analysis of the results.
        output_file_type="txt"        # The type of the Analysis of the results.
    )

    # Step2: Get the circuits group.
    circuits_list = benchmark.get_circuits(
        quantum_machine_info={"qubits_number": 3, "layout_file": layout, "Instruction_Set": Inset},
        # The sub-physical machine properties to be measured.
        # qubits_number is The number of physical machine bits, layout_file is Physical machine topology,
        # Instruction_Set is Physical machine instruction set type.
        level=3,              # choose circuits group level, one of [1, 2, 3].
        mapping=True,         # Mapping according to the physical machine topology or not.
        gate_transform=True   # Gate transform according to the physical machine Instruction Set or not.
    )
    print(len(circuits_list))

    # Here the sub-physical machine to be measured is simulated.
    amp_results_list = []
    for circuit in circuits_list:
        amp_results_list.append(simulation(circuit))

    # Step3: Enter the evalute system
    benchmark.evaluate(circuits_list, amp_results_list)

    print(os.listdir("./benchmark"))


def Run():
    # Second method: Initialize -> Connect physical machines to perform benchmarks.

    # Step1: initialize QuICTBenchmark.
    benchmark = QuICTBenchmark(
        device="CPU",                 # choose simulation device, one of [CPU, GPU]
        output_path="./benchmark",    # The path of the Analysis of the results.
        output_file_type="txt"        # The type of the Analysis of the results.
    )

    # Step2: Connect physical machines to perform benchmarking.
    benchmark.run(
        simulator_interface=simulation,  # Perform simulation of amplitude for each circuit.
        quantum_machine_info={"qubits_number": 3, "layout_file": layout, "Instruction_Set": Inset},
        level=3,
        mapping=True,
        gate_transform=True
    )
    print(os.listdir("./benchmark"))


if __name__ == "__main__":
    Step_benchmark()
    Run()
