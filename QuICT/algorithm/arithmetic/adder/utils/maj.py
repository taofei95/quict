from QuICT.core.gate import CompositeGate
from QuICT.core.gate import CCX, CX


class Maj(CompositeGate):
    def __init__(self, name: str = "MAJ"):
        super().__init__(name)

        with self:
            CX & [2, 1]
            CX & [2, 0]
            CCX & [0, 1, 2]


class UnMaj(CompositeGate):
    def __init__(self, name: str = "UMA"):
        super().__init__(name)

        with self:
            CCX & [0, 1, 2]
            CX & [2, 0]
            CX & [0, 1]
