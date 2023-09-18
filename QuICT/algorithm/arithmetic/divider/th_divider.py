from QuICT.algorithm.arithmetic.adder import TRIOCarryAdder, MuThCtrlAdder
from QuICT.core.gate import CompositeGate, X, CX
from QuICT.tools.exception.core.gate_exception import GateParametersAssignedError


class SubtractionModule(CompositeGate):
    """
        Implement a quantum subtractor named subtraction
        which is a module of the following divider.

        Input:
            The first "n" qubits are subtrahend of which the higher digit in lower lines.
            The following "n" qubits are minuend of which the higher digit in lower lines.
            For both subtrahend and minuend, the highest digit must set to |0>
            due to 2’s complement positive binary.

        Output:
            The first "n" qubits are the result of "b-a"
            The following "n" qubits are reserved for the minuend.
    """

    def __init__(
            self,
            qreg_size: int,
            name: str = None
    ):
        """
            Construct the subtraction module calculates ~(~b+a) which is equivalent to b-a.
            For both subtrahend and minuend, the highest digit must set to |0>
            due to 2’s complement positive binary.

            |b>|a> --->|b-a>|a>

            Circuit width: '2 * qreg_size'.
            Args:
                qreg_size (int): The input quantum register size for subtrahend and minuend.
                name (str): The name of subtraction module. Default to None.
        """
        if qreg_size < 3:
            raise GateParametersAssignedError(
                f"Register size must be greater than or equal to 3 but given {qreg_size}"
            )
        super().__init__(name)

        self._qreg_a_list = list(range(qreg_size, 2 * qreg_size))
        self._qreg_b_list = list(range(qreg_size))

        # calculate ~b
        for i in self._qreg_b_list:
            X | self([i])

        # apply TRIOCarryAdder
        adder_apply_list = self._qreg_b_list.copy()
        adder_apply_list.extend(self._qreg_a_list[1::])
        adder_apply_list.append(self._qreg_a_list[0])
        TRIOCarryAdder(qreg_size - 1) | self(adder_apply_list)

        # calculate ~(~b+a)
        for i in self._qreg_b_list:
            X | self([i])


class CtrlAddSubModule(CompositeGate):
    """
        Implement a quantum adder-subtractor circuit named Ctrl-AddSub
        which is a module of following divider.

        Input:
            The first qubit is the "control" qubit which determine
            whether the module is an adder or subtractor.
            The following "n" qubits are first operand of which the higher digit in lower lines.
            The last "n" qubits are second operand of which the higher digit in lower lines.
            For both operands, the highest digit must set to |0>
            due to 2’s complement positive binary.

        Output:
            The first qubit id reserved for 'Ctrl' qubit.
            The following "n" qubits are the result of "b-a" or "b+a".
            The last "n" qubits are reserved for the operand.
    """

    def __init__(
            self,
            qreg_size: int,
            name: str = None
    ):
        """
            Construct the Ctrl-AddSub Module which calculates "a + b"
            when "ctrl" input is low. Otherwise, calculates "b - a".
            For both operands, the highest digit must set to |0>
            due to 2’s complement positive binary.

            If ctrl is |0>:
            |ctrl>|b>|a> ---> |ctrl>|b+a>|a>
            If ctrl is |1>
            |ctrl>|b>|a> ---> |ctrl>|b-a>|a>

            Circuit width: '2 * qreg_size + 1'.
            Args:
                qreg_size (int): The input quantum register size for operands.
                name (str): The name of CtrlAddSUbModule. Default to None.
        """
        if qreg_size < 3:
            raise GateParametersAssignedError(
                f"Register size must be greater than or equal to 3 but given {qreg_size}"
            )
        super().__init__(name)

        self._ctrl_qubit = [0]
        self._qreg_a_list = list(range(qreg_size + 1, 2 * qreg_size + 1))
        self._qreg_b_list = list(range(1, qreg_size + 1))

        # ctrl-calculate ~b
        for i in self._qreg_b_list:
            CX | self([self._ctrl_qubit[0], i])

        # apply TRIOCarryAdder
        adder_apply_list = self._qreg_b_list.copy()
        adder_apply_list.extend(self._qreg_a_list[1::])
        adder_apply_list.append(self._qreg_a_list[0])
        TRIOCarryAdder(qreg_size - 1) | self(adder_apply_list)

        # ctrl-calculate ~(~b+a)
        for i in self._qreg_b_list:
            CX | self([self._ctrl_qubit[0], i])


class CtrlAddNopModule(CompositeGate):
    """
        Implement a quantum conditional addition circuit called Ctrl_AddNop
        which is a module of following divider.

        Input:
            The first qubit is the "control" qubit which determine whether to do add.
            The following "n" qubits are first addend of which the higher digit in lower lines.
            The last "n" qubits are second addend of which the higher digit in lower lines.
            For both addends, the highest digit must set to |0>
            due to 2’s complement positive binary.

        Output:
            The first qubit id reserved for 'Ctrl' qubit.
            The following "n" qubits are the result of "b + a" or reserved for the first addend.
            The last "n" qubits are reserved for the second addend.
    """

    def __init__(
            self,
            qreg_size: int,
            name: str = None
    ):
        """
            Construct the Ctrl-AddNop Module which calculates "a + b"
            when "ctrl" input is high. Otherwise, do nothing.
            For both addends, the highest digit must set to |0>
            due to 2’s complement positive binary.

            |ctrl>|b>|a> ---> |ctrl>|(ctrl * a) + b>|a>

            Circuit width: '2 * qreg_size + 1'.
            Args:
                qreg_size (int): The input quantum register size for addends.
                name (str): The name of CtrlAddNopModule. Default to None.
        """
        if qreg_size < 3:
            raise GateParametersAssignedError(
                f"Register size must be greater than or equal to 3 but given {qreg_size}"
            )
        super().__init__(name)

        self._ctrl_qubit = [0]
        self._qreg_a_list = list(range(qreg_size + 1, 2 * qreg_size + 1))
        self._qreg_b_list = list(range(1, qreg_size + 1))

        # apply MuThCtrlAdder
        adder_apply_list = self._ctrl_qubit.copy()
        adder_apply_list.extend(self._qreg_a_list[1::])
        adder_apply_list.extend(self._qreg_b_list)
        adder_apply_list.append(self._qreg_a_list[0])
        MuThCtrlAdder(qreg_size - 1) | self(adder_apply_list)


class THRestoreDivider(CompositeGate):
    """
        Implement a divider using Restoring Division Algorithm.

        Input:
            The first 'n' qubits are set to |0>.
            The following 'n' qubits are the dividend of which the higher digit in lower lines.
            The last 'n' qubits are the divisor of which the higher digit in lower lines.

        Output:
            The first 'n' qubits are the quotient.
            The following 'n' qubits are the remainder.
            The last 'n' qubits are reserved for divisor

        Based on paper "Quantum Circuit Designs of Integer Division Optimizing T-count and T-depth"
        by Himanshu Thapliyal, Edgard Muñoz-Coreas, T.S.S.Varun and Travis S.Humble[1]

        [1]: https://ieeexplore.ieee.org/document/8691552
    """

    def __init__(self, qreg_size, name: str = None):
        """
            Construct a divider using Restoring Division Algorithm, which calculates "b/a" and
            stores quotient and remainder. For both operands, the highest digit must set to |0>
            due to 2’s complement positive binary.

            |0...0>|b>|a> ---> |b//a>|b%a>|a>

            Circuit width: '3 * qreg_size'.
            Args:
                qreg_size (int): The input quantum register size for divisor and dividend.
                name (str): The name of CtrlAddNopModule. Default to None.
        """
        if qreg_size < 3:
            raise GateParametersAssignedError(
                f"Register size must be greater than or equal to 3 but given {qreg_size}"
            )
        super().__init__(name)

        self._qreg_q_list = list(range(qreg_size))
        self._qreg_r_list = list(range(qreg_size, 2 * qreg_size))
        self._qreg_a_list = list(range(2 * qreg_size, 3 * qreg_size))

        for i in range(qreg_size):
            iteration = self._qreg_q_list[i::].copy()
            iteration.extend(self._qreg_r_list[:i + 1:])
            iteration.extend(self._qreg_a_list)
            self._build_normal_iteration(qreg_size) | self(iteration)

    def _build_normal_iteration(self, qreg_size) -> CompositeGate:
        """
            Construct the circuit generation of iteration of quantum restoring division circuit.
        """
        iteration = CompositeGate()

        # step 1
        SubtractionModule(qreg_size) | iteration(list(range(1, 2 * qreg_size + 1)))
        # step 2
        CX | iteration([1, 0])
        # step 3
        CtrlAddNopModule(qreg_size) | iteration(list(range(2 * qreg_size + 1)))
        # step 4
        X | iteration(0)

        return iteration


class THNonRestDivider(CompositeGate):
    """
        Implement a divider using Non-Restoring Division Algorithm.

        Input:
            The first 'n - 1' qubits are set to |0>.
            The following 'n' qubits are the dividend of which the higher digit in lower lines.
            The last 'n' qubits are the divisor of which the higher digit in lower lines.

        Output:
            The first 'n' qubits are the quotient.
            The following 'n - 1' qubits are the remainder.
            The last 'n' qubits are reserved for divisor

        Based on paper "Quantum Circuit Designs of Integer Division Optimizing T-count and T-depth"
        by Himanshu Thapliyal, Edgard Muñoz-Coreas, T.S.S.Varun and Travis S.Humble[1]

        [1]: https://ieeexplore.ieee.org/document/8691552
    """

    def __init__(self, qreg_size: int, name: str = None):
        """
            Construct a divider using Non-Restoring Division Algorithm, which calculates "b/a" and
            stores quotient and remainder. For both operands, the highest digit must set to |0>
            due to 2’s complement positive binary.

            |0...bn-1>|bn-1...b0>|a> ---> |b//a>|b%a>|a>

            Circuit width: '3 * qreg_size - 1'.
            Args:
                qreg_size (int): The input quantum register size for divisor and dividend.
                name (str): The name of CtrlAddNopModule. Default to None.
        """
        if qreg_size < 3:
            raise GateParametersAssignedError(
                f"Register size must be greater than or equal to 3 but given {qreg_size}"
            )
        super().__init__(name)

        self._qreg_q_list = list(range(qreg_size))
        self._qreg_r_list = list(range(qreg_size, 2 * qreg_size - 1))
        self._qreg_a_list = list(range(2 * qreg_size - 1, 3 * qreg_size - 1))

        # step 1
        sub_list = self._qreg_q_list.copy()
        sub_list.extend(self._qreg_a_list)
        SubtractionModule(qreg_size) | self(sub_list)

        # step 2
        for i in range(qreg_size - 1):
            iteration = self._qreg_q_list[i::].copy()
            iteration.extend(self._qreg_r_list[:i + 1:])
            iteration.extend(self._qreg_a_list)
            X | self([self._qreg_q_list[i]])
            CtrlAddSubModule(qreg_size) | self(iteration)

        # step 3
        add_nop_list = [self._qreg_q_list[qreg_size - 1]]
        add_nop_list.extend(self._qreg_r_list)
        add_nop_list.extend(self._qreg_a_list[1::])
        CtrlAddNopModule(qreg_size - 1) | self(add_nop_list)
        X | self([self._qreg_q_list[qreg_size - 1]])
