import os

from QuICT.benchmark import QuantumMachinebenchmark
from QuICT.core.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.core.virtual_machine import InstructionSet, VirtualQuantumMachine
from QuICT.simulation.state_vector import StateVectorSimulator


layout = Layout.linear_layout(5)
iset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])
vqm = VirtualQuantumMachine(
    qubits=5,
    instruction_set=iset,
    layout=layout,
)


def simulation(circuit):
    """Simulate the quantum physical machine interface.

    Args:
        circuit: A quantum circuit.

    Returns:
        List: Amplitude results or Sample results for circuits.
    """
    simulator = StateVectorSimulator()
    simulator.run(circuit)
    sim_results = simulator.sample()
    return sim_results


def step_benchmark():
    # First method: Initialize -> Get the circuits -> Results Analysis.

    # Step1: initialize QuICTBenchmark.
    benchmark = QuantumMachinebenchmark(
        output_path="./benchmark",    # The path of the Analysis of the results.
        output_file_type="txt"        # The type of the Analysis of the results.
    )

    # Step2: Get the circuits group.
    circuits_list = benchmark.get_circuits(
        quantum_machine_info=vqm,       # The information about the quantum machine - virual quantum machine.
        level=1,                        # choose circuits group level, one of [1, 2, 3].
        enable_qcda_for_alg_cir=True,   # Auto-Compile the algorithm circuit with the given quantum machine info,
        is_measure=True                 # can choose whether to measure the circuit according to your needs.
    )
    print(len(circuits_list))

    # Here the sub-physical machine to be measured is simulated.
    for circuit in circuits_list:
        circuit.machine_amp = simulation(circuit.circuit)

    # Step3: Enter the evalute system
    benchmark.show_result(circuits_list)

    print(os.listdir("./benchmark"))


def run():
    # Second method: Initialize -> Connect physical machines to perform benchmarks.

    # Step1: initialize QuICTBenchmark.
    benchmark = QuantumMachinebenchmark(
        output_path="./benchmark",    # The path of the Analysis of the results.
        output_file_type="txt"        # The type of the Analysis of the results.
    )

    # Step2: Connect physical machines to perform benchmarking.
    benchmark.run(
        simulator_interface=simulation,  # Perform simulation of amplitude for each circuit.
        quantum_machine_info=vqm,
        level=3,
        enable_qcda_for_alg_cir=False,
        is_measure=False
    )
    print(os.listdir("./benchmark"))


if __name__ == "__main__":
    run()
