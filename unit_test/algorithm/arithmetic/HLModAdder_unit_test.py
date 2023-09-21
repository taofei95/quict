import unittest
import numpy as np

from QuICT.tools import Logger
from QuICT.tools.exception.core import CircuitAppendError
from QuICT.core import Circuit
from QuICT.core.gate.gate import X
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.arithmetic.adder.hl_adder import HLModAdder

HLModAdder_logger = Logger("test_HLModAdder_gates(adder)")


class TestHLModAdderGates(unittest.TestCase):
    """ Test the Quantum Modular Adder Gates named HLModAdder. """

    def test_HLModAdder_unsigned(self):
        """
            Test using the HLModAdder gates to do unsigned modular addition on two qregs.
        """

        # regular case
        self.assertEqual(self._get_HLModAdder_result(5, 8, 14), 8 + 14)
        # one qubit case
        self.assertEqual(self._get_HLModAdder_result(1, 0, 1), 0 + 1)
        self.assertEqual(self._get_HLModAdder_result(1, 0, 0), 0 + 0)
        self.assertEqual(self._get_HLModAdder_result(1, 1, 0), 1 + 0)
        self.assertEqual(self._get_HLModAdder_result(1, 1, 1), (1 + 1) % 2)
        # two qubits case
        self.assertEqual(self._get_HLModAdder_result(2, 1, 2), 1 + 2)
        # overflow
        self.assertEqual(self._get_HLModAdder_result(5, 16, 26), (16 + 26) % (2 ** 5))
        # overflow of twp qubits case
        self.assertEqual(self._get_HLModAdder_result(2, 3, 2), (3 + 2) % 4)
        # random inputs of two qubits case
        a, b = np.random.randint(4, size=2)
        self.assertEqual(self._get_HLModAdder_result(2, a, b), (a + b) % 4)
        # random inputs
        qreg_size = np.random.randint(1, 11)
        x, y = np.random.randint(2 ** qreg_size, size=2)
        self.assertEqual(
            self._get_HLModAdder_result(qreg_size, x, y),
            (x + y) % (2 ** qreg_size)
        )

    def test_HLModAdder_signed(self):
        """
             Test using the HLModAdder gates to do signed module]ar addition on two qregs.
        """

        # regular case
        self.assertEqual(self._get_HLModAdder_result(5, 1, 1, signed=True), 1 + 1)
        self.assertEqual(self._get_HLModAdder_result(5, 8, -2, signed=True), 8 - 2)
        self.assertEqual(self._get_HLModAdder_result(5, -16, 4, signed=True), -16 + 4)
        self.assertEqual(self._get_HLModAdder_result(5, -3, -6, signed=True), -3 - 6)
        # one qubit case
        self.assertEqual(self._get_HLModAdder_result(1, 0, 0, signed=True), 0 + 0)
        self.assertEqual(self._get_HLModAdder_result(1, 0, -1, signed=True), 0 - 1)
        self.assertEqual(self._get_HLModAdder_result(1, -1, 0, signed=True), -1 + 0)
        # two qubits case
        self.assertEqual(self._get_HLModAdder_result(2, 0, 1, signed=True), 1)
        self.assertEqual(self._get_HLModAdder_result(2, 0, -1, signed=True), -1)
        self.assertEqual(self._get_HLModAdder_result(2, -2, 1, signed=True), -1)
        self.assertEqual(self._get_HLModAdder_result(2, -1, -1, signed=True), -2)
        # overflow
        self.assertEqual(
            self._get_HLModAdder_result(5, 12, 9, signed=True),
            (12 + 9 + 2 ** (5 - 1)) % (2 ** 5) - 2 ** (5 - 1)
        )
        # overflow of two qubits case
        self.assertEqual(
            self._get_HLModAdder_result(2, 1, 1, signed=True),
            (1 + 1 + 2) % 4 - 2
        )
        # underflow
        self.assertEqual(
            self._get_HLModAdder_result(5, -16, -13, signed=True),
            (-16 - 13 + 2 ** (5 - 1)) % (2 ** 5) - 2 ** (5 - 1)
        )
        # underflow of one qubit
        self.assertEqual(
            self._get_HLModAdder_result(1, -1, -1, signed=True),
            0
        )
        # underflow of two qubits
        self.assertEqual(
            self._get_HLModAdder_result(2, -2, -1, signed=True),
            (-2 - 1 + 2) % 4 - 2
        )
        # random inputs of two qubits case
        a, b = np.random.randint(-2, 2, size=2)
        self.assertEqual(
            self._get_HLModAdder_result(2, a, b, signed=True),
            (a + b + 2) % 4 - 2
        )
        # random inputs
        qreg_size = np.random.randint(1, 11)
        x, y = np.random.randint(-2 ** (qreg_size - 1), 2 ** (qreg_size - 1), size=2)
        self.assertEqual(
            self._get_HLModAdder_result(qreg_size, x, y, signed=True),
            (x + y + 2 ** (qreg_size - 1)) % (2 ** qreg_size) - 2 ** (qreg_size - 1)
        )

    def _get_HLModAdder_result(
        self,
        qreg_size: int,
        q_x: int,
        q_y: int,
        signed: bool = False
    ) -> int:
        """
            Args:
                qreg_size (int): size of the quantum register.
                q_x (int): integer encode into quantum register.
                q_y (int): integer encode into quantum register.
                signed (int): if True, will add one more bit for sign.

            Returns:
                "q_x + q_y mod 2**qreg_size" by running the adder circuit and decoding the result.
        """

        try:
            q_x_bin = np.binary_repr(q_x, qreg_size)
        except CircuitAppendError as e:
            HLModAdder_logger.error(e("Not enough register size to hold q_x"))
            raise
        try:
            q_y_bin = np.binary_repr(q_y, qreg_size)
        except CircuitAppendError as e:
            HLModAdder_logger.error(e("Not enough register size to hold q_y"))

        # Construct Circuit #

        # init circuit with q_x and q_y
        mod_adder_cir = Circuit(2 * qreg_size)
        for i, bit in enumerate(q_x_bin):
            if bit == '1':
                X | mod_adder_cir([i])
        for i, bit in enumerate(q_y_bin, start=qreg_size):
            if bit == '1':
                X | mod_adder_cir([i])

        # apply HLModAdder
        HLModAdder(qreg_size) | mod_adder_cir

        # Decode
        sv_sim = StateVectorSimulator()
        sv_sim.run(circuit=mod_adder_cir)
        # only the correct answer will have non-zero amplitude
        result = sv_sim.sample(shots=1, target_qubits=list(range(0, qreg_size)))
        for idx, val in enumerate(result):
            if val != 0:
                if signed:
                    return self._unsigned_to_signed(idx, qreg_size)
                else:
                    return idx

    def _unsigned_to_signed(self, value: int, bit_len: int):
        return value - (value >> (bit_len - 1)) * (2 ** bit_len)


if __name__ == "__main__":
    unittest.main()
