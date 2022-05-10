from ._operator import Operator


class Multiply(Operator):
    def __init__(self, value):
        super().__init__(targets=1)
        self._value = value

    @property
    def value(self):
        return self._value
