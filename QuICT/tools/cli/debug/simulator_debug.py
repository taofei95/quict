from QuICT.core import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.simulation.density_matrix import DensityMatrixSimulator
from QuICT.simulation.unitary import UnitarySimulator
from .circuit_generator import generate_random_circuit_by_type


def simulator_test(backend, device, qubits: int = 4, test_gate: str = "single", seed: int = 2023):
    circuit = generate_random_circuit_by_type(
        test_gate, qubits, seed
    )

    if backend == "unitary":
        simulator = UnitarySimulator(device)

        return simulator.run(circuit)
    elif backend == "state_vector":
        sv_test(device, circuit)
    elif backend == "density_matrix":
        dm_test(device, circuit)


def sv_test(device, circuit: Circuit):
    simulator = StateVectorSimulator(device)

    # Step by gate
    simulator.initial_circuit(circuit)
    simulator.initial_state_vector()
    gidx = 0
    for gate, qidx, _ in circuit.fast_gates:
        print(f"Gate Indexes: {gidx}")
        simulator._apply_gate(gate, qidx)
        print(simulator.vector())


def dm_test(device, circuit: Circuit):
    simulator = DensityMatrixSimulator(device)

    # Step by gate
    simulator.initial_circuit(circuit)
    simulator._gate_calculator.get_allzero_density_matrix(circuit.width())

    gidx = 0
    for gate, qidx, _ in circuit.fast_gates:
        print(f"Gate Indexes: {gidx}")
        simulator.apply_gates([gate & qidx])

        print(simulator._density_matrix)
