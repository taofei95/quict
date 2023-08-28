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

    def run(self, circuit: str, shots: int = 1000):
        prog, _, _ = convert_qasm_to_qprog(circuit, self.qm)

        # Call quantum chip
        result = self.qm.real_chip_measure(prog, shots)

        self.qm.finalize()
        quafu_amp = [0] * (2 ** circuit.width())
        for key, value in result.items():
            quafu_amp[int(key, 2)] = value
        amp_result = np.array(quafu_amp)

        return amp_result
