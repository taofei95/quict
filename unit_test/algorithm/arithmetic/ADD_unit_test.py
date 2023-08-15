import unittest
import numpy as np

from QuICT.tools import Logger
from QuICT.tools.exception.core import CircuitAppendError
from QuICT.core import Circuit
from QuICT.core.gate.gate import X
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.arithmetic.adder.quantum_adder import ADD

ADD_logger = Logger("test_ADD_gates(adder)")


class TestADDGates(unittest.TestCase):
    """Test the Quantum Adder Gates named ADD"""

    def test_ADD_unsigned(self):
        """
            Test using the ADD gates to do unsigned addition on two qregs
            and reserve the carry qubit in the result.
        """

        # regular case
        self.assertEqual(self._get_ADD_result(5, 8, 14), 8 + 14)
        # one qubit case
        self.assertEqual(self._get_ADD_result(1, 1, 1), 1 + 1)
        # random inputs of one qubit case
        x, y = np.random.randint(2, size=2)
        self.assertEqual(self._get_ADD_result(1, x, y), x + y)
        # random inputs
        qreg_size = np.random.randint(1, 11)
        x, y = np.random.randint(2 ** qreg_size, size=2)
        self.assertEqual(self._get_ADD_result(qreg_size, x, y), x + y)

    def test_ADD_signed(self):
        """
            Test using the ADD gates to do signed addition on two qregs
            and ignore the carry qubit in the result.
        """

        # regular case
        self.assertEqual(
            self._get_ADD_result(5, 6, 7, signed=True),
            6 + 7
        )
        self.assertEqual(
            self._get_ADD_result(5, 15, -4, signed=True),
            15 - 4
        )
        self.assertEqual(
            self._get_ADD_result(5, -8, 14, signed=True),
            -8 + 14
        )
        self.assertEqual(
            self._get_ADD_result(5, -2, -3, signed=True),
            -2 - 3
        )
        # one qubit case
        self.assertEqual(
            self._get_ADD_result(1, 0, -1, signed=True),
            0 - 1
        )
        self.assertEqual(
            self._get_ADD_result(1, -1, 0, signed=True),
            -1 - 0
        )
        self.assertEqual(self._get_ADD_result(1, 0, 0, signed=True), 0)
        # overflow
        self.assertEqual(
            self._get_ADD_result(5, 15, 7, signed=True),
            (15 + 7 + 2 ** (5 - 1)) % (2 ** 5) - 2 ** (5 - 1)
        )
        # underflow
        self.assertEqual(
            self._get_ADD_result(5, -7, -13, signed=True),
            (-7 - 13 + 2 ** (5 - 1)) % (2 ** 5) - 2 ** (5 - 1)
        )
        # underflow of one qubit case
        self.assertEqual(self._get_ADD_result(1, -1, -1, signed=True), 0)
        # random inputs
        reg_size = np.random.randint(1, 11)
        x, y = np.random.randint(-2 ** (reg_size - 1), 2 ** (reg_size - 1), size=2)
        self.assertEqual(
            self._get_ADD_result(reg_size, x, y, signed=True),
            (x + y + 2 ** (reg_size - 1)) % (2 ** reg_size) - 2 ** (reg_size - 1)
        )

    def test_ADD_error_case(self):
        """
            Test using the ADD gates to do addition on two qregs
            with the first qubit being set to |1> by mistake,
            and the carry bit will be reversed.
        """

        self.assertEqual(
            self._get_ADD_error_input_result(5, 1, 3),
            (1 + 3) ^ (1 << 5)
        )
        # random inputs
        qreg_size = np.random.randint(1, 11)
        x, y = np.random.randint(2 ** qreg_size, size=2)
        self.assertEqual(
            self._get_ADD_error_input_result(qreg_size, x, y),
            (x + y) ^ (1 << qreg_size)
        )

    def _get_ADD_result(
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
                "q_x + q_y" by running the adder circuit and decoding the result.
        """

        try:
            q_x_bin = np.binary_repr(q_x, qreg_size)
        except CircuitAppendError as e:
            ADD_logger.error(e("Not enough register size to hold q_x"))
            raise
        try:
            q_y_bin = np.binary_repr(q_y, qreg_size)
        except CircuitAppendError as e:
            ADD_logger.error(e("Not enough register size to hold q_y"))

        # Construct Circuit #

        # init circuit with q_x and q_y
        adder_circ = Circuit(2 * qreg_size + 1)
        for i, bit in enumerate(q_x_bin, start=1):
            if bit == '1':
                X | adder_circ([i])
        for i, bit in enumerate(q_y_bin, start=qreg_size + 1):
            if bit == '1':
                X | adder_circ([i])

        # apply ADD
        ADD(qreg_size) | adder_circ

        # Decode #
        sv_sim = StateVectorSimulator()
        sv_sim.run(circuit=adder_circ)
        # only the correct answer will have non-zero amplitude
        if signed:
            result = sv_sim.sample(shots=1, target_qubits=list(range(1, qreg_size + 1)))
            for idx, val in enumerate(result):
                if val != 0:
                    return self._unsigned_to_signed(idx, qreg_size)
        else:
            result = sv_sim.sample(shots=1, target_qubits=list(range(0, qreg_size + 1)))
            for idx, val in enumerate(result):
                if val != 0:
                    return idx

    def _get_ADD_error_input_result(
        self,
        qreg_size: int,
        q_x: int,
        q_y: int,
    ) -> int:
        """
            Test error case when first qubit is set to |1>.

            Args:
                qreg_size (int): size of the quantum register.
                q_x (int): integer encode into quantum register.
                q_y (int): integer encode into quantum register.

            Return:
                "q_x + q_y" by running the adder circuit in above error case
                and decoding the result.
        """

        try:
            q_x_bin = np.binary_repr(q_x, qreg_size)
        except CircuitAppendError as e:
            ADD_logger.error(e("Not enough register size to hold q_x"))
            raise
        try:
            q_y_bin = np.binary_repr(q_y, qreg_size)
        except CircuitAppendError as e:
            ADD_logger.error(e("Not enough register size to hold q_y"))

        # init circuit with q_x and q_y
        adder_circ = Circuit(2 * qreg_size + 1)
        for i, bit in enumerate(q_x_bin, start=1):
            if bit == '1':
                X | adder_circ([i])
        for i, bit in enumerate(q_y_bin, start=qreg_size + 1):
            if bit == '1':
                X | adder_circ([i])

        # set error case
        X | adder_circ([0])

        # apply ADD
        ADD(qreg_size) | adder_circ

        # Decode #
        sv_sim = StateVectorSimulator()
        sv_sim.run(circuit=adder_circ)
        # only the correct answer will have non-zero amplitude
        result = sv_sim.sample(shots=1, target_qubits=list(range(0, qreg_size + 1)))
        for idx, val in enumerate(result):
            if val != 0:
                return idx

    def _unsigned_to_signed(self, value: int, bit_len: int):
        return value - (value >> (bit_len - 1)) * (2 ** bit_len)


if __name__ == '__main__':
    unittest.main()
