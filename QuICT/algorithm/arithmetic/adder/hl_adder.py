from QuICT.core.gate import CX, CompositeGate
from .utils import HLPeres, HLTR2


class HLCarryAdder(CompositeGate):
    r"""
        A ripple-carry in-place half adder that in total requries "2n + 1" qubits. For two n-qubit binary
        encoded integers `a` and `b`, calculates their sum and store on the first "n + 1" qubits:

        $$
            \vert{0}\rangle\vert{a}\rangle_n\vert{b}\rangle_n \to \vert{a+b}\rangle_{n+1}\vert{b}\rangle_n
        $$

        Applying this adder on two 3-qubit sized register, $a:=\vert{q_1q_2q_3}\rangle$ and
        $b:=\vert{q_4q_5q_6}\rangle$ with a clean qubit $\vert{q_0}\rangle$ as output carry looks like:

                          ┌────┐                                    ┌──────────┐           »
            q_0: |0>──────┤ cx ├────────────────────────────────────┤2         ├───────────»
                    ┌────┐└─┬──┘                                    │          │           »
            q_1: |0>┤ cx ├──┼───────────────────────────────────────┤1         ├───────────»
                    └─┬──┘  │   ┌────┐                  ┌──────────┐│          │┌─────────┐»
            q_2: |0>──┼─────┼───┤ cx ├──────────────────┤1         ├┤  cg_Pere ├┤0        ├»
                      │     │   └─┬──┘      ┌──────────┐│          ││          ││         │»
            q_3: |0>──┼─────┼─────┼─────────┤1         ├┤          ├┤          ├┤         ├»
                      │     │     │   ┌────┐│          ││  cg_Pere ││          ││  cg_TR2 │»
            q_4: |0>──■─────■─────┼───┤ cx ├┤          ├┤2         ├┤0         ├┤2        ├»
                                  │   └─┬──┘│  cg_Pere ││          │└──────────┘│         │»
            q_5: |0>──────────────■─────■───┤2         ├┤0         ├────────────┤1        ├»
                                            │          │└──────────┘            └─────────┘»
            q_6: |0>────────────────────────┤0         ├───────────────────────────────────»
                                            └──────────┘                                   »
            «
            «q_0: ─────────────────────────────────────────
            «                                        ┌────┐
            «q_1: ───────────────────────────────────┤ cx ├
            «                                  ┌────┐└─┬──┘
            «q_2: ─────────────■───────────────┤ cx ├──┼───
            «     ┌─────────┐  │               └─┬──┘  │
            «q_3: ┤0        ├──┼─────■───────────┼─────┼───
            «     │         │  │     │   ┌────┐  │     │
            «q_4: ┤         ├──┼─────┼───┤ cx ├──┼─────■───
            «     │  cg_TR2 │┌─┴──┐  │   └─┬──┘  │
            «q_5: ┤2        ├┤ cx ├──┼─────■─────■─────────
            «     │         │└────┘┌─┴──┐
            «q_6: ┤1        ├──────┤ cx ├──────────────────
            «     └─────────┘      └────┘

        Examples:
            >>> from QuICT.core import Circuit
            >>> from QuICT.algorithm.arithmetic import HLCarryAdder
            >>>
            >>> circuit = Circuit(7)
            >>> HLCarryAdder(3) | circuit

        Note:
            The first qubit must be set to $\vert{0}\rangle$ to get the right output carry qubit in the result.
            If the first qubit is $\vert{1}\rangle$, the carry qubit in the result will be reversed.

        !!! Note "Implementation Details(Asymptotic)"

            | Parameter    | Info                  |
            | ------------ | --------------------- |
            | Input Size   | $n$                   |
            | Input carry  | 0                     |
            | Output carry | 1                     |
            | num. ancilla | 0                     |
            | Gate set     | $CX, H, T, T^\dagger$ |
            | Width        | $2n + 1$              |
            | Depth        | $16n-3$               |
            | Size         | $35n-21$              |
            | CX count     | $17n-12$              |

        References:
            [1]: "Design of Efficient Reversible Logic-Based Binary and BCD Adder Circuits"
            by Himanshu Thapliyal and Nagarajan Ranganayhan <https://arxiv.org/pdf/1712.02630.pdf>.

            [2]: "Efficient quantum arithmetic operation circuits for quantum image processing"
            by Hai-Sheng Li, Ping Fan, Haiying Xia, Huiling Peng and Gui-Lu Long
            <https://doi.org/10.1007/s11433-020-1582-8>.
    """

    def __init__(self, figures: int, name: str = None):
        """
            Construct the adder circuit that compute the sum of two quantum registers
            of size "figures".

            Args:
                figures (int): The qubits' figures in addend.
                name (str, optional): The name of HLCarryAdder gates. Defaults to None.

            Raises:
                GateParametersAssignedError: If `figures` is smaller than 1.
        """
        assert figures >= 1, "HLCarryAdder Gate need at least one figures in addend."
        super().__init__(name)
        self.add_build(figures)

    def add_build(self, figures):
        pg1_gate = HLPeres()
        tr2_gate = HLTR2()
        CX | self([1 + figures, 1])
        CX | self([1 + figures, 0])
        for i in range(2 + figures, 2 * figures):
            CX | self([i, i - figures])
            CX | self([i, i - 1])
        for i in range(2 * figures, figures + 1, -1):
            pg1_gate | self([i, i - figures, i - 1])
        pg1_gate | self([figures + 1, 1, 0])
        for i in range(1 + figures, 2 * figures):
            tr2_gate | self([i - figures + 1, i + 1, i])
        for i in range(2, figures + 1):
            CX | self([i, i + figures])
        for i in range(2 * figures - 1, figures + 1, -1):
            CX | self([i, i - 1])
            CX | self([i, i - figures])
        CX | self([1 + figures, 1])


class HLModAdder(CompositeGate):
    r"""
        A ripple-carry in-place adder that in total requires "2n" qubits. For two n-qubit binary
        encoded integers `a` and `b`, calculates their sum modulus 2^n and store on the first "n qubits:

        $$
            \vert{a}\rangle_n\vert{b}\rangle_n \to \vert{a+b \mod 2^n}\rangle_{n}\vert{b}\rangle_n
        $$

        Applying this adder on two 3-qubit sized register, $a:=\vert{q_0q_1q_2}\rangle$ and
        $b:=\vert{q_3q_4q_5}\rangle$ looks like:

                    ┌────┐      ┌────┐            ┌──────────┐
            q_0: |0>┤ cx ├──────┤ cx ├────────────┤2         ├───────────────────────
                    └─┬──┘┌────┐└─┬──┘            │          │                 ┌────┐
            q_1: |0>──┼───┤ cx ├──┼───────────────┤1         ├─────────────────┤ cx ├
                      │   └─┬──┘  │   ┌──────────┐│          │┌─────────┐      └─┬──┘
            q_2: |0>──┼─────┼─────┼───┤1         ├┤  cg_Pere ├┤0        ├──■─────┼───
                      │     │     │   │          ││          ││         │  │     │
            q_3: |0>──■─────┼─────┼───┤          ├┤          ├┤         ├──┼─────┼───
                            │     │   │  cg_Pere ││          ││  cg_TR2 │  │     │
            q_4: |0>────────■─────■───┤2         ├┤0         ├┤2        ├──┼─────■───
                                      │          │└──────────┘│         │┌─┴──┐
            q_5: |0>──────────────────┤0         ├────────────┤1        ├┤ cx ├──────
                                      └──────────┘            └─────────┘└────┘

        Examples:
            >>> from QuICT.core import Circuit
            >>> from QuICT.algorithm.arithmetic import HLModAdder
            >>>
            >>> circuit = Circuit(6)
            >>> HLModAdder(3) | circuit

        !!! Note "Implementation Details(Asymptotic)"

            | Parameter    | Info                  |
            | ------------ | --------------------- |
            | Input Size   | $n$                   |
            | num. ancilla | 0                     |
            | Gate set     | $CX, H, T, T^\dagger$ |
            | Width        | $2n$                  |
            | Depth        | $16n-19$              |
            | Size         | $35n-55$              |
            | CX count     | $17n-28$              |

        References:
            [1]: "Design of Efficient Reversible Logic-Based Binary and BCD Adder Circuits"
            by Himanshu Thapliyal and Nagarajan Ranganayhan <https://arxiv.org/pdf/1712.02630.pdf>.

            [2]: "Efficient quantum arithmetic operation circuits for quantum image processing"
            by Hai-Sheng Li, Ping Fan, Haiying Xia, Huiling Peng and Gui-Lu Long
            <https://doi.org/10.1007/s11433-020-1582-8>.
    """

    def __init__(self, figures: int, name: str = None):
        """
            Construct the modular adder circuit.

            Args:
                figures (int): The qubits' figures in addend.
                name (str, optional): The name of HLModAdder gates. Defaults to None.

            Raises:
                GateParametersAssignedError: If `figures` is smaller than 1.
        """
        assert figures >= 1, "HLModAdder Gate need at least one figures in addend."
        super().__init__(name)
        self.madd_build(figures)

    def madd_build(self, figures):
        pg1_gate = HLPeres()
        tr2_gate = HLTR2()
        if figures == 1:
            CX | self([1, 0])
        elif figures == 2:
            CX | self([2, 0])
            CX | self([3, 1])
        else:
            for i in range(0, figures - 1):
                CX | self([i + figures, i])
        if figures >= 2:
            CX | self([figures + 1, 0])
        for i in range(figures + 1, 2 * figures - 2):
            CX | self([i + 1, i])
        for i in range(2 * figures - 1, figures + 1, -1):
            pg1_gate | self([i, i - figures, i - 1])
        if figures >= 2:
            pg1_gate | self([figures + 1, 1, 0])
        for i in range(figures + 1, 2 * figures - 1):
            tr2_gate | self([i - figures + 1, i + 1, i])
        for i in range(2, figures):
            CX | self([i, i + figures])
        for i in range(2 * figures - 2, figures + 1, -1):
            CX | self([i, i - 1])
            CX | self([i, i - figures])
        if figures >= 2:
            CX | self([1 + figures, 1])
