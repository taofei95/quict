from collections import defaultdict
import os
import numpy as np

from QuICT.core.utils import unique_id_generator


class Result:
    """ Data structure class for simulator result

    Args:
        mode (str): The simulator mode, usually be {device-backend}.
        shots (int): The running times; must be a positive integer.
        options (dict): Optional parameters for the simulator.
    """
    def __init__(
        self,
        device: str,
        backend: str,
        shots: int,
        options: dict
    ):
        self.id = unique_id_generator()
        self.device = device
        self.backend = backend
        self.shots = shots
        self.options = options
        self.spending_time = 0
        self.counts = defaultdict(int)

        # prepare output path
        self.output_path = self._prepare_output_file()

    def _prepare_output_file(self):
        """ Prepare output path. """
        curr_path = os.getcwd()
        output_path = os.path.join(curr_path, "output", self.id)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        return output_path

    def record(self, result, spending_time: float = None, qubits: int = None):
        """ Record circuit's result

        Args:
            result (int or np.array): The measured result from StateVectorSimulator or matrix from Unitary Simulator.

        Raises:
            TypeError: Wrong type input.
        """
        if self.backend == "unitary":
            self.unitary_matrix = result
        elif self.backend == "statevector":
            bit_idx = "{0:0b}".format(result)
            if qubits:
                bit_idx = bit_idx.zfill(qubits)
            self.counts[bit_idx] += 1

        if spending_time is not None:
            self.spending_time += spending_time / self.shots

    def record_circuit(self, circuit):
        """ dump the circuit. """
        with open(f"{self.output_path}/circuit.qasm", "w") as of:
            of.write(circuit.qasm())

    def record_sv(self, state, shot):
        """ dump the circuit. """
        if self.device == "GPU" and self.backend == "statevector":
            state = state.get()

        np.savetxt(f"{self.output_path}/state_{shot}.txt", state)

    def dumps(self):
        """ dump the result. """
        with open(f"{self.output_path}/result.log", "w") as of:
            of.write(str(self.__dict__))

        return self.__dict__
