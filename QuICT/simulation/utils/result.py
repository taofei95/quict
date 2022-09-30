import os
import numpy as np


class Result:
    """ Data structure class for simulator result

    Args:
        mode (str): The simulator mode, usually be {device-backend}.
        shots (int): The running times; must be a positive integer.
        options (dict): Optional parameters for the simulator.
    """
    def __init__(
        self,
        circuit_id: str,
        device: str,
        backend: str,
        shots: int,
        options: dict
    ):
        self.id = circuit_id
        self.device = device
        self.backend = backend
        self.shots = shots
        self.options = options
        self.counts = {}
        self.state_vector = None
        self.density_matrix = None

        # prepare output path
        self.output_path = self._prepare_output_file()

    def __str__(self):
        return f"ID: {self.id}\nDevice: {self.device}\nBackend: {self.backend}\nShots: {self.shots}\n" + \
            f"Options: {self.options}\nResults: {self.counts}"

    def __dict__(self):
        return {
            "id": self.id,
            "shots": self.shots,
            "backend_info": {
                "device": self.device,
                "backend": self.backend,
                "options": self.options
            },
            "data": {
                "counts": self.counts,
                "state_vector": self.state_vector,
                "density_matrix": self.density_matrix
            }
        }

    def _prepare_output_file(self):
        """ Prepare output path. """
        curr_path = os.getcwd()
        output_path = os.path.join(curr_path, "output", self.id)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        return output_path

    def record_sample(self, result: list):
        """ Record circuit's result

        Args:
            result (list): The sample of measured result from given circuit.
        """
        for i in range(len(result)):
            bit_idx = "{0:0b}".format(i)
            bit_idx = bit_idx.zfill(int(np.log2(len(result))))
            self.counts[bit_idx] = result[i]

        with open(f"{self.output_path}/result.log", "w") as of:
            of.write(str(self.__dict__))

    def record_circuit(self, circuit):
        """ dump the circuit. """
        with open(f"{self.output_path}/circuit.qasm", "w") as of:
            of.write(circuit.qasm())

    def record_amplitude(self, amplitude, is_record: bool = False):
        """ dump the circuit. """
        if self.device == "GPU":
            amplitude = amplitude.get()

        if is_record:
            np.savetxt(f"{self.output_path}/amplitude.txt", amplitude)

        if self.backend == "density_matrix":
            self.density_matrix = amplitude.copy()
        else:
            self.state_vector = amplitude.copy()
