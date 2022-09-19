import sys

from QuICT.simulation import Simulator
from QuICT.tools.interface import OPENQASMInterface
from utils import local_redis_set


def simulation_start(
    circuit_path: str,
    shots: int,
    device: str,
    backend: str,
    precision: str,
    output_path: str
):
    # Get circuit from given path
    circuit = OPENQASMInterface.load_file(circuit_path).circuit

    # Start Simulator
    simulator = Simulator(
        shots=shots,
        device=device,
        backend=backend,
        precision=precision,
        output_path=output_path
    )
    simulator.run(circuit)


if __name__=="__main__":
    name = sys.argv[1]
    local_redis_set(name)
    simulation_start(
        sys.argv[2],
        int(sys.argv[3]),
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
        sys.argv[7]
    )
