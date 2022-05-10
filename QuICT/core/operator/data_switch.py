from enum import Enum

from ._operator import Operator


class DataSwitchType(Enum):
    all = "ALL"
    half = "HALF"
    ctarg = "CTARGS"
    prob = "PROB_ADD"


class DataSwitch(Operator):
    def __init__(self, destination: int, type: DataSwitchType, switch_condition: dict = None):
        super().__init__(targets = 1)
        self._destination = destination
        self._type = type
        self._cond = switch_condition

    @property
    def destination(self):
        return self._destination

    @property
    def type(self):
        return self._type

    @property
    def switch_condition(self):
        return self._cond
