from QuICT.core.gate import H, CX, T, T_dagger, CompositeGate


class HLPeres(CompositeGate):
    r"""
        Implement a Peres Gate using T gates:

        $$
            \vert{a}\rangle \vert{b}\rangle \vert{c}\rangle
            \to
            \vert{a}\rangle \vert{a \oplus b}\rangle \vert{a \cdot b\oplus c}\rangle
        $$

        References:
            [1]: "Efficient quantum arithmetic operation circuits for quantum image processing" by
            Hai-Sheng Li, Ping Fan, Haiying Xia, Huiling Peng and Gui-Lu Long
            <https://doi.org/10.1007/s11433-020-1582-8>
    """

    def __init__(self, name: str = None):
        """
            Args:
                 name (str): the name of the Peres gate. Default to None.
        """
        super().__init__(name)
        self.pg_build()

    def pg_build(self):
        """
            Construct a Peres Gate.
        """
        H | self([2])
        CX | self([1, 2])
        CX | self([2, 0])
        T_dagger | self([0])
        T_dagger | self([1])
        T | self([2])
        CX | self([1, 0])
        T | self([0])
        CX | self([1, 2])
        CX | self([2, 0])
        CX | self([0, 1])
        T_dagger | self([0])
        T | self([1])
        T_dagger | self([2])
        H | self([2])
