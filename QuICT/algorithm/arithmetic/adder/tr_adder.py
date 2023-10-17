from QuICT.core.gate import CompositeGate, CX, CCX, X
from .utils import HLPeres, HLTR1
from QuICT.tools.exception.core.gate_exception import GateParametersAssignedError


class TRIOCarryAdder(CompositeGate):
    r"""
        A ripple-carry in-place full adder that in total requires "2n + 2" qubits. For two
        n-qubit binary encoded integers `a` and `b`, calculates their sum and store it on the
        first "n + 1" qubits:

        $$
            \vert{0}\rangle \vert{a}\rangle_n \vert{b}\rangle_n \vert{c_{in}}\rangle
            \to
            \vert{a + b}\rangle_{n+1} \vert{b}\rangle_n \vert{c_{in}}\rangle
        $$

        Applying this adder on two 3-qubit sized register, $a:=\vert{q_1q_2q_3}\rangle$ and
        $b:=\vert{q_4q_5q_6}\rangle$ with $\vert{q_7}\rangle$ as input carry and $\vert{q_0}\rangle$
        as output carry looks like:

                                                        ┌────┐                        »
            q_0: |0>────────────────────────────────────┤ cx ├────────────────────────»
                    ┌────┐                              └─┬──┘                        »
            q_1: |0>┤ cx ├────────────────────────────────┼───────────────────────────»
                    └─┬──┘┌────┐                          │                      ┌───┐»
            q_2: |0>──┼───┤ cx ├──────────────────────────┼──────────────────■───┤ x ├»
                      │   └─┬──┘┌────┐                    │          ┌───┐   │   └───┘»
            q_3: |0>──┼─────┼───┤ cx ├────────────────────┼──────■───┤ x ├───┼────────»
                      │     │   └─┬──┘                    │      │   └───┘   │        »
            q_4: |0>──■─────┼─────┼─────────────────■─────■──────┼───────────┼────────»
                            │     │               ┌─┴──┐         │        ┌──┴──┐     »
            q_5: |0>────────■─────┼───────────■───┤ cx ├─────────┼────────┤ ccx ├─────»
                                  │         ┌─┴──┐└────┘      ┌──┴──┐     └──┬──┘     »
            q_6: |0>──────────────■─────■───┤ cx ├────────────┤ ccx ├────────■────────»
                                      ┌─┴──┐└────┘            └──┬──┘                 »
            q_7: |0>──────────────────┤ cx ├─────────────────────■────────────────────»
                                      └────┘                                          »
            «     ┌──────────┐
            «q_0: ┤2         ├──────────────────────────────────────────────
            «     │          │                            ┌────┐
            «q_1: ┤1         ├────────────────────────────┤ cx ├────────────
            «     │          │┌─────────┐   ┌───┐         └─┬──┘┌────┐
            «q_2: ┤          ├┤1        ├───┤ x ├───────────┼───┤ cx ├──────
            «     │  cg_Pere ││         │┌──┴───┴──┐┌───┐   │   └─┬──┘┌────┐
            «q_3: ┤          ├┤         ├┤1        ├┤ x ├───┼─────┼───┤ cx ├
            «     │          ││         ││         │└───┘   │     │   └─┬──┘
            «q_4: ┤          ├┤  cg_TR1 ├┤         ├──■─────■─────┼─────┼───
            «     │          ││         ││         │┌─┴──┐        │     │
            «q_5: ┤0         ├┤2        ├┤  cg_TR1 ├┤ cx ├──■─────■─────┼───
            «     └──────────┘│         ││         │└────┘┌─┴──┐        │
            «q_6: ────────────┤0        ├┤2        ├──────┤ cx ├──■─────■───
            «                 └─────────┘│         │      └────┘┌─┴──┐
            «q_7: ───────────────────────┤0        ├────────────┤ cx ├──────
            «                            └─────────┘            └────┘

        Examples:
            >>> from QuICT.core import Circuit
            >>> from QuICT.algorithm.arithmetic import TRIOCarryAdder
            >>>
            >>> circuit = Circuit(8)
            >>> TRIOCarryAdder(3) | circuit

        Note:
            The first qubit, must be set to |0> to get the right output carry qubit in the result.
            If the first qubit is |1>, the carry qubit in the result will be reversed.

        !!! Note "Implementation Details(Asymptotic)"

            | Parameter    | Info                          |
            | ------------ | ----------------------------- |
            | Input Size   | $n$                           |
            | Input carry  | $1$                           |
            | Output carry | $1$                           |
            | num. ancilla | $0$                           |
            | Gate set     | $CCX, CX, X, H, T, T^\dagger$ |
            | Width        | $2n+2$                        |
            | Depth        | $10n+5$                       |
            | Size         | $22n -2$                      |
            | CX count     | $10n + 1$                     |
            | CCX count    | $n-1$                         |

        References:
            [1]: "Design of Efficient Reversible Logic-Based Binary and BCD Adder Circuits"
            by Himanshu Thapliyal and Nagarajan Ranganayhan <https://arxiv.org/pdf/1712.02630.pdf>.
    """

    def __init__(
        self,
        qreg_size: int,
        name: str = None
    ):
        """
            Construct a reversible ripple carry full adder.

            Args:
                qreg_size (int): The qubits figures for both addend.
                name (str, optional): The name of TRIOCarryAdder gates. Defaults to None.

            Raises:
                GateParametersAssignedError: If `qreg_size` is smaller than 2.
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
