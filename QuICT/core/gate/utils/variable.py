import numpy as np
from typing import Union
import uuid


class Variable:
    @property
    def pargs(self):
        return self._pargs

    @property
    def grads(self):
        return self._grads

    @property
    def identity(self):
        return self._identity

    @property
    def shape(self):
        return self._shape

    @pargs.setter
    def pargs(self, pargs):
        if isinstance(pargs, np.ndarray):
            self._pargs = pargs.astype(np.float64)
        else:
            self._pargs = np.float64(pargs)

    @grads.setter
    def grads(self, grads):
        if isinstance(grads, np.ndarray):
            assert grads.shape == self._pargs.shape
            self._grads = grads.astype(np.float64)
        else:
            self._grads = np.float64(grads)

    @identity.setter
    def identity(self, uid):
        if isinstance(uid, str):
            self._identity = uid

    def __init__(
        self,
        pargs: Union[float, np.float64, np.ndarray],
        grads: Union[float, np.float64, np.ndarray] = None,
        identity: str = None,
    ):
        if isinstance(pargs, np.ndarray):
            self._pargs = pargs.astype(np.float64)
        else:
            self._pargs = np.float64(pargs)
        if grads is None:
            self._grads = np.zeros(pargs.shape, dtype=np.float64)
        else:
            if isinstance(grads, np.ndarray):
                assert grads.shape == pargs.shape
                self._grads = grads.astype(np.float64)
            else:
                self._grads = np.float64(grads)
        self._shape = self._pargs.shape
        self._identity = (
            identity if identity is not None else str(uuid.uuid1()).replace("-", "")
        )

    def __getitem__(self, index):
        return Variable(
            pargs=self.pargs[index],
            grads=self.grads[index],
            identity=self.identity + str(index).replace("(", "").replace(")", ""),
        )

    def __eq__(self, other):
        if (self.pargs == other.pargs).all() and (self.grads == other.grads).all():
            return True
        else:
            return False

    def __add__(self, other):
        if isinstance(other, (int, float, np.float64)):
            if isinstance(self.pargs, np.float64):
                return Variable(
                    pargs=self.pargs + other, grads=self.grads, identity=self.identity
                )
        raise TypeError

    def __sub__(self, other):
        if isinstance(other, (int, float, np.float64)):
            if isinstance(self.pargs, np.float64):
                return Variable(
                    pargs=self.pargs - other, grads=self.grads, identity=self.identity
                )
        raise TypeError

    def __mul__(self, other):
        if isinstance(other, (int, float, np.float64)):
            if isinstance(self.pargs, np.float64):
                grads = other if abs(self.grads) < 1e-12 else self.grads * other
                return Variable(
                    pargs=self.pargs * other, grads=grads, identity=self.identity
                )
        raise TypeError


if __name__ == "__main__":
    pargs = Variable(np.array([[0, 1, 2], [0.1, 0.2, 0.3]]))
    print(pargs[0, 1].pargs)
    print((pargs[0, 1] * 3 + 2).pargs)
    print(pargs.pargs)
    print(pargs.identity)
    print(pargs[0, 1].identity)
    # print((pargs[0, 1] * 3 + 2).identity)
    
    s = "1"
    t = tuple(map(int, s.split(', ')))
    print(t)
    a = np.array([1,2,3])
    print(a[t])

