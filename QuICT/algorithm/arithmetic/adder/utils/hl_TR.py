from QuICT.core.gate import H, CX, T, T_dagger, CompositeGate


class HLTR1(CompositeGate):
    """
        Implement a TR1 Gate using T gates:

        |a>|b>|c> ---> |a>|a⊕b>|a.~b⊕c>

        Based on papers ""Efficient quantum arithmetic operation circuits for quantum image processing"
        by Hai-Sheng Li, Ping Fan, Haiying Xia, Huiling Peng and Gui-Lu Long:
        https://doi.org/10.1007/s11433-020-1582-8
    """

    def __init__(self, name: str = None):
        """
            Args:
                name (str, optional): The name of TR1 gate. Defaults to None.
        """
        super().__init__(name)
        self.tr1_build()

    def tr1_build(self):
        """
            Construct a TR1 Gate.
        """
        H | self([2])
        CX | self([1, 2])
        CX | self([2, 0])
        T | self([2])
        T_dagger | self([1])
        T_dagger | self([0])
        CX | self([1, 0])
        CX | self([1, 2])
        T_dagger | self([0])
        CX | self([2, 0])
        CX | self([0, 1])
        T | self([2])
        T | self([1])
        T | self([0])
        H | self([2])


class HLTR2(CompositeGate):
    """
        Implement a TR2 Gate using T gates:

        |a>|b>|c> ---> |a>|a⊕b>|~a.b⊕c>

        Based on papers ""Efficient quantum arithmetic operation circuits for quantum image processing"
        by Hai-Sheng Li, Ping Fan, Haiying Xia, Huiling Peng and Gui-Lu Long:
        https://doi.org/10.1007/s11433-020-1582-8
    """

    def __init__(self, name: str = None):
        """
            Args:
                name (str, optional): The name of TR2 gate. Defaults to None.
        """
        super().__init__(name)
        self.tr2_build()

    def tr2_build(self):
        """
            Construct a TR2 Gate.
        """
        T | self([0])
        T_dagger | self([1])
        H | self([2])
        CX | self([0, 1])
        T_dagger | self([2])
        CX | self([1, 2])
        CX | self([2, 0])
        T | self([0])
        T_dagger | self([1])
        T | self([2])
        CX | self([1, 0])
        T_dagger | self([0])
        CX | self([1, 2])
        CX | self([2, 0])
        H | self([2])
