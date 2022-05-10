
from typing import List

from QuICT.core.utils import unique_id_generator
from ._operator import Operator


class CheckPoint(Operator):
    @property
    def uid(self):
        return self._uid

    @property
    def position(self):
        return self._pos

    @position.setter
    def position(self, shift: int):
        assert isinstance(shift, int)
        self._pos += shift

    def __init__(self):
        super().__init__(targets=1)
        self._uid = unique_id_generator()
        self._pos = -1

    def get_child(self, shift: int = 0):
        assert isinstance(shift, int)
        return CheckPointChild(self.uid, shift)

    def match(self, uid: str):
        return uid == self._uid

    def __or__(self, targets):
        super().__or__(targets)
        self._pos = targets.size()


class CheckPointChild(Operator):
    @property
    def uid(self):
        return self._uid

    @property
    def shift(self):
        return self._shift

    def __init__(self, uid: str, shift: int = 0):
        super().__init__(targets=1)
        self._uid = uid
        self._shift = shift

    def find_position(self, checkpoints: List[CheckPoint]):
        pos = -1
        for cp in checkpoints:
            if cp.match(self._uid):
                pos = cp.position
                cp.position = self._shift

        if pos == -1:
            raise ValueError("Cannot find parent checkpoint for current checkpointchild.") 

        return pos
