from QuICT.core.gate import H, CX, T, T_dagger, CompositeGate


class PG1(CompositeGate):
    """ Implement a PG1 Gate which is a component of the following adders. """

    def __init__(self, name: str = None):
        """
        Args:
            name (str, optional): The name of PG1 gate. Defaults to None.
        """
        super().__init__(name)
        self.pg1_build()

    def pg1_build(self):
        H | self(0)
        CX | self([1, 0])
        CX | self([0, 2])
        T | self(0)
        T_dagger | self(1)
        T_dagger | self(2)
        CX | self([1, 2])
        CX | self([1, 0])
        T | self(2)
        CX | self([0, 2])
        CX | self([2, 1])
        T_dagger | self(0)
        T | self(1)
        T_dagger | self(2)
        H | self(0)


class TR2(CompositeGate):
    """ Implement a TR2 Gate which is a component of the following adders. """

    def __init__(self, name: str = None):
        """
        Args:
            name (str, optional): The name of TR2 gate. Defaults to None.
        """
        super().__init__(name)
        self.tr2_build()

    def tr2_build(self):
        H | self(0)
        T_dagger | self(0)
        T | self(1)
        T_dagger | self(2)
        CX | self([1, 2])
        CX | self([2, 0])
        CX | self([0, 1])
        T | self(0)
        T | self(1)
        T_dagger | self(2)
        CX | self([2, 1])
        T_dagger | self(1)
        CX | self([2, 0])
        CX | self([0, 1])
        H | self(0)


class HLCarryAdder(CompositeGate):
    """
        Implement a quantum-quantum adder circuits using in total "2n + 1" qubits.

        Input:
            The first qubit must be set to |0> to get the right carry qubit in the result.
            If the first qubit is |1>, the carry qubit in the result will be reversed.
            The following "n" qubits are the first addend and
            the last "n" qubits are the second addend.

        Output:
            The first "n + 1" qubits are the sum in which the first qubit is the carry qubit.
            The last "n" qubits are reserved for the second addend.

        Based on papers "Design of Efficient Reversible Logic-Based Binary and BCD Adder Circuits"
        by Himanshu Thapliyal and Nagarajan Ranganayhan: https://arxiv.org/pdf/1712.02630.pdf
        and "Efficient quantum arithmetic operation circuits for quantum image processing"
        by Hai-Sheng Li, Ping Fan, Haiying Xia, Huiling Peng and Gui-Lu Long:
        https://doi.org/10.1007/s11433-020-1582-8
    """

    def __init__(self, figures: int, name: str = None):
        """
            Construct the adder circuit that compute the sum of two quantum registers
            of size "figures".

            Args:
                figures (int): The qubits' figures in addend.
                name (str, optional): The name of HLCarryAdder gates. Defaults to None.
        """
        assert figures >= 1, "HLCarryAdder Gate need at least one figures in addend."
        super().__init__(name)
        self.add_build(figures)

    def add_build(self, figures):
        pg1_gate = PG1()
        tr2_gate = TR2()
        CX | self([1 + figures, 1])
        CX | self([1 + figures, 0])
        for i in range(2 + figures, 2 * figures):
            CX | self([i, i - figures])
            CX | self([i, i - 1])
        for i in range(2 * figures, figures + 1, -1):
            pg1_gate | self([i - 1, i - figures, i])
        pg1_gate | self([0, 1, figures + 1])
        for i in range(1 + figures, 2 * figures):
            tr2_gate | self([i, i - figures + 1, i + 1])
        for i in range(2, figures + 1):
            CX | self([i, i + figures])
        for i in range(2 * figures - 1, figures + 1, -1):
            CX | self([i, i - 1])
            CX | self([i, i - figures])
        CX | self([1 + figures, 1])


class HLModAdder(CompositeGate):
    """
        Implement a quantum-quantum modular adder circuits Gates using in total "2n" qubits.

        Input:
            The first "n" qubits are the first addend.
            The last "n" qubits are the second addend.
        Output:
            The first "n" qubits are the sum.
            The last "n" qubits are reserved for the second addend.

        Based on paper "Efficient quantum arithmetic operation circuits for quantum image processing"
        by Hai-Sheng Li, Ping Fan, Haiying Xia, Huiling Peng and Gui-Lu Long:
        https://doi.org/10.1007/s11433-020-1582-8
    """

    def __init__(self, figures: int, name: str = None):
        """
            Construct the modular adder circuit that does:

            s = (a + b) mod 2^n

            a and b are two number stored in quantum register of size 'figures'
            n = 'figures'

            Args:
                figures (int): The qubits' figures in addend.
                name (str, optional): The name of HLModAdder gates. Defaults to None.
        """
        assert figures >= 1, "HLModAdder Gate need at least one figures in addend."
        super().__init__(name)
        self.madd_build(figures)

    def madd_build(self, figures):
        pg1_gate = PG1()
        tr2_gate = TR2()
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
            pg1_gate | self([i - 1, i - figures, i])
        if figures >= 2:
            pg1_gate | self([0, 1, figures + 1])
        for i in range(figures + 1, 2 * figures - 1):
            tr2_gate | self([i, i - figures + 1, i + 1])
        for i in range(2, figures):
            CX | self([i, i + figures])
        for i in range(2 * figures - 2, figures + 1, -1):
            CX | self([i, i - 1])
            CX | self([i, i - figures])
        if figures >= 2:
            CX | self([1 + figures, 1])
