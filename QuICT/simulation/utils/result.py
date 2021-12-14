from collections import defaultdict
import os
import uuid


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
        self.output_path = self._prepare_output_file()

    def _generate_uuid(self):
        """ Generate unique ID for result. """
        u_id = uuid.uuid1()
        u_id = str(u_id).replace("-", "")
        return u_id

    def _prepare_output_file(self):
        """ Prepare output path. """
        curr_path = os.getcwd()
        output_path = os.path.join(curr_path, "output")

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        return output_path

    def record(self, result):
        """ Record circuit's result

        Args:
            result (dict or [Qubits]): The qubits after the simulator run or the result dict from remote simulator.

        Raises:
            TypeError: Wrong type input.
        """
        if isinstance(result, dict):
            self.counts = result
        elif isinstance(result, list):
            result_idx = 0
            for idx, qubit in enumerate(result):
                if qubit.measured:
                    result_idx += 1 << idx

            bit_idx = "{0:0b}".format(result_idx)
            bit_idx = bit_idx.zfill(len(result))
            self.counts[bit_idx] += 1
        else:
            raise TypeError("Only recore qubits' state and result from remote simulator.")

    def dumps(self):
        """ dump the result. """
        with open(f"{self.output_path}/{self.id}.log", "w") as of:
            of.write(str(self.__dict__))

        return self.__dict__
