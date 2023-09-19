from QuICT.core.gate import CompositeGate, CX, CCX, X
from .utils import HLPeres, HLTR1
from QuICT.tools.exception.core.gate_exception import GateParametersAssignedError


class TRIOCarryAdder(CompositeGate):
    """
        Implement a quantum-quantum reversible ripple carry adder with input carry
        using `2 * n + 2` qubits.

        Input:
            The first qubit must be set to |0> to get the right carry qubit in the result.
            If the first qubit is |1>, the carry qubit in the result will be reversed.
            The following "n" qubits are the first addend.
            The next "n" qubits are the second addend.
            The last qubit is the input carry.
        Output:
            The first "n + 1" qubits are the sum in which the first qubit is the carry qubit.
            The following "n" qubits are reserved for the second addend.
            The last qubit is reserved for the input carry.

        Based on paper "Design of Efficient Reversible Logic-Based Binary and BCD Adder Circuits"
        by Himanshu Thapliyal and Nagarajan Ranganayhan:
        https://arxiv.org/pdf/1712.02630.pdf
    """

    def __init__(
        self,
        qreg_size: int,
        name: str = None
    ):
        """
            Construct a reversible ripple carry adder with input carry which
            calculates "s = a + b + c0" and stores in b's qreg:

            |0>|b>|a>|c0> ---> |sn>|sn-1...s0>|a>|c0>

            Args:
                qreg_size (int): The qubits figures for both addend.
                name (str, optional): The name of TRIOCarryAdder gates. Defaults to None.
        """
        if qreg_size < 2:
            raise GateParametersAssignedError(f"Register size must be greater than or equal to 2 but given {qreg_size}")
        super().__init__(name)

        self._out_carry = [0]
        self._b_list = list(range(1, qreg_size + 1))
        self._a_list = list(range(qreg_size + 1, 2 * qreg_size + 1))
        self._in_carry = [2 * qreg_size + 1]

        # step 1
        for i in range(0, qreg_size):
            CX | self([self._a_list[i], self._b_list[i]])

        # step 2
        CX | self([self._a_list[qreg_size - 1], self._in_carry[0]])
        for i in range(qreg_size - 1, 0, -1):
            CX | self([self._a_list[i - 1], self._a_list[i]])
        CX | self([self._a_list[0], self._out_carry[0]])

        # step 3
        CCX | self([
            self._b_list[qreg_size - 1],
            self._in_carry[0],
            self._a_list[qreg_size - 1]
        ])
        X | self([self._b_list[qreg_size - 1]])
        for i in range(qreg_size - 2, 0, -1):
            CCX | self([self._b_list[i], self._a_list[i + 1], self._a_list[i]])
            X | self([self._b_list[i]])
        HLPeres() | self([self._a_list[1], self._b_list[0], self._out_carry[0]])

        # step 4
        for i in range(1, qreg_size - 1):
            HLTR1() | self([
                self._a_list[i + 1],
                self._b_list[i],
                self._a_list[i]
            ])
            X | self([self._b_list[i]])
        HLTR1() | self([
            self._in_carry[0],
            self._b_list[qreg_size - 1],
            self._a_list[qreg_size - 1]
        ])
        X | self([self._b_list[qreg_size - 1]])

        # step 5
        for i in range(0, qreg_size - 1):
            CX | self([self._a_list[i], self._a_list[i + 1]])
        CX | self([self._a_list[qreg_size - 1], self._in_carry[0]])

        # step 6
        for i in range(0, qreg_size):
            CX | self([self._a_list[i], self._b_list[i]])
