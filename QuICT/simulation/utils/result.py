from collections import defaultdict
import os
import uuid
import numpy as np


# TODO: Add circuit and state vector output. Add spending time
class Result:
    """ Data structure class for simulator result

    Args:
        mode (str): The simulator mode, usually be {device-backend}.
        shots (int): The running times; must be a positive integer.
        options (dict): Optional parameters for the simulator.
    """
    def __init__(
        self,
        mode: str,
        shots: int,
        options: dict
    ):
        self.id = self._generate_uuid()
        self.mode = mode
        self.shots = shots
        self.options = options
        self.counts = defaultdict(int)
        self._spending_time = 0
        self.output_path = self._prepare_output_file()

    def _generate_uuid(self):
        """ Generate unique ID for result. """
        u_id = uuid.uuid1()
        u_id = str(u_id).replace("-", "")
        return u_id

    def _prepare_output_file(self):
        """ Prepare output path. """
        curr_path = os.getcwd()
        output_path = os.path.join(curr_path, "output", self.id)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        return output_path

    def record(self, result: int, qubits: int = None):
        """ Record circuit's result

        Args:
            result (dict or int): The final state of circuit or the result dict from remote simulator.

        Raises:
            TypeError: Wrong type input.
        """
        if isinstance(result, dict):
            self.counts = result
        elif isinstance(result, int):
            bit_idx = "{0:0b}".format(result)
            if qubits:
                bit_idx = bit_idx.zfill(qubits)
            self.counts[bit_idx] += 1
        else:
            raise TypeError("Only recore qubits' state and result from remote simulator.")

    def record_time(self, spending_time: float):
        self._spending_time += spending_time / self.shots

    def record_circuit(self, circuit):
        """ dump the circuit. """
        with open(f"{self.output_path}/circuit.qasm", "w") as of:
            of.write(circuit.qasm())

    def record_sv(self, state, shot):
        """ dump the circuit. """
        if type(state) is not np.ndarray:
            state = state.get()

        np.savetxt(f"{self.output_path}/state_{shot}.txt", state)

    def dumps(self):
        """ dump the result. """
        with open(f"{self.output_path}/result.log", "w") as of:
            of.write(str(self.__dict__))

        return self.__dict__
