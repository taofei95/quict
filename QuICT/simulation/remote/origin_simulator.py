# ----------------------------
# Origin Quantum
# https://qcloud.originqc.com.cn/
# ----------------------------
import numpy as np

from pyqpanda import *


class OriginSimulator:
    def __init__(self, token: str):
        # Create Virtual Quantum Machine through QCloud
        self.qm = QCloud()

        # Initialize
        self.qm.init_qvm(token)

    def run(self, circuit: str, shots: int = 1000, chip_id: int = 2):
        prog, _, _ = convert_qasm_to_qprog(circuit, self.qm)

        assert chip_id in list(range(1, 4))
        # Call quantum chip
        result = self.qm.real_chip_measure(prog, shots, chip_id)
        self.qm.finalize()
        origin_amp = [0] * (2 ** circuit.width())
        for key, value in result.items():
            origin_amp[int(key, 2)] = value
        amp_result = np.array(origin_amp)

        return amp_result
