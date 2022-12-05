from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation import Simulator


def simulate_random_circuit(device, backend, shots, precision):
    cir = Circuit(5)
    cir.random_append(20)

    sim = Simulator(device, backend, precision)
    result = sim.run(cir, shots)

    print(result["data"])


if __name__ == "__main__":
    device = "GPU"                  # one of [CPU, GPU], warning make sure have a GPU Environment, if using GPU device
    backend = "density_matrix"        # one of [state_vector, unitary, density_matrix]
    shots = 1000                    # The time of sample.
    precision = "single"            # one of [single, double]

    simulate_random_circuit(device, backend, shots, precision)
