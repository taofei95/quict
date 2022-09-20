import sys

from QuICT.simulation import Simulator
from QuICT.tools.interface import OPENQASMInterface


def simulation_start(
    circuit_path: str,
    shots: str,
    device: str,
    backend: str,
    precision: str,
    output_path: str
):
    # Get circuit from given path
    circuit = OPENQASMInterface.load_file(circuit_path).circuit

    # Start Simulator
    simulator = Simulator(
        shots=int(shots),
        device=device,
        backend=backend,
        precision=precision,
        output_path=output_path
    )
    simulator.run(circuit)


if __name__ == "__main__":
    simulation_start(
        *sys.argv[1:]
    )
