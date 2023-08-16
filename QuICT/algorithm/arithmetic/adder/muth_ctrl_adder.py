from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.core.gate import CX, CCX

from QuICT.tools.exception.core.gate_exception import GateParametersAssignedError


class MuThCtrlAdder(CompositeGate):
    """
        A conditional addition with no input carry and no garbage outputs proposed by
        Edgard Muñoz-Coreas and Himanshu Thapliyal in "Quantum Circuit Design of a T-count
        Optimized Integer Multiplier"[1]

        [1]: https://ieeexplore.ieee.org/document/8543237
    """

    def __init__(
        self,
        qreg_size: int,
        name: str = None
    ):
        """
            Contruct a toffoli based controlled in-place adder that based on the control bit,
            calculates 's = a + b' and store on b's register. This adder requires two ancilla
            qubits. One of the ancilla qubits is for possible carry-out when calculating 's'.
            :

            |ctrl>|a>|0>|b>|0> ---> |ctrl>|a>|(ctrl * a) + b>|0>

            Total circuit requires `2 * qreg_size + 3` qubits.

            Args:
                qreg_size (int): Input register size for both addends. >= 2.

        """
        if qreg_size < 2:
            raise GateParametersAssignedError("Register size must be greater than or equal to 2.")

        self._reg_size = qreg_size

        self._ctrl_bit = [0]
        self._reg_a_list = list(range(1, 1 + qreg_size))
        self._reg_carry = [1 + qreg_size]
        self._reg_b_list = list(range(qreg_size + 2, 2 * qreg_size + 2))
        self._anci_list = [2 * qreg_size + 2]

        super().__init__(name)

        # step 1
        for i in range(qreg_size - 1):
            CX | self([self._reg_a_list[i], self._reg_b_list[i]])

        # step 2
        CCX | self(self._ctrl_bit + [self._reg_a_list[0]] + self._reg_carry)
        for i in range(qreg_size - 2):
            CX | self([self._reg_a_list[i + 1], self._reg_a_list[i]])

        # step 3
        for i in range(qreg_size - 1):
            CCX | self([
                self._reg_b_list[-1 - i],
                self._reg_a_list[-1 - i],
                self._reg_a_list[-2 - i]
            ])

        # step 4
        CCX | self([self._reg_b_list[0], self._reg_a_list[0]] + self._anci_list)
        CCX | self(self._ctrl_bit + self._anci_list + self._reg_carry)
        CCX | self([self._reg_b_list[0], self._reg_a_list[0]] + self._anci_list)

        # step 5
        for i in range(qreg_size - 1):
            CCX | self(self._ctrl_bit + [self._reg_a_list[i], self._reg_b_list[i]])
            CCX | self([
                self._reg_b_list[i + 1],
                self._reg_a_list[i + 1],
                self._reg_a_list[i]
            ])
        CCX | self(self._ctrl_bit + [self._reg_a_list[-1], self._reg_b_list[-1]])

        # step 6
        for i in range(qreg_size - 2):
            CX | self([self._reg_a_list[-2 - i], self._reg_a_list[-3 - i]])

        # step 7
        for i in range(qreg_size - 1):
            CX | self([self._reg_a_list[i], self._reg_b_list[i]])
