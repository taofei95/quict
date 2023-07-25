import numpy as np
from typing import Union
import uuid


class Variable(object):
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
    def index(self):
        index = self.identity[32:]
        return tuple(map(int, index.split(", ")[:-1]))

    @property
    def shape(self):
        return self._shape

    @property
    def origin_shape(self):
        return self._origin_shape

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
        origin_shape: tuple = None,
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
        self._origin_shape = self._shape if origin_shape is None else origin_shape
        self._identity = (
            identity if identity is not None else str(uuid.uuid1()).replace("-", "")
        )

    def __getitem__(self, index):
        return Variable(
            pargs=self.pargs[index],
            grads=self.grads[index],
            identity=self.identity
            + str(index).replace("(", "").replace(")", "")
            + ", ",
            origin_shape=self.origin_shape,
        )

    def __eq__(self, other):
        if (self.pargs == other.pargs).all() and (self.grads == other.grads).all():
            return True
        else:
            return False

    def __add__(self, other):
        if isinstance(other, (int, float, np.float64, np.ndarry)):
            if isinstance(other, np.ndarry) and other.shape != self.pargs.shape:
                raise ValueError
            return Variable(
                pargs=self.pargs + other,
                grads=self.grads,
                identity=self.identity,
                origin_shape=self.origin_shape,
            )
        raise TypeError

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return self.__mul__(-1.0).__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.float64)):
            if isinstance(self.pargs, np.float64):
                grads = other if abs(self.grads) < 1e-12 else self.grads * other
            else:
                grads = self.grads * other
                grads[abs(grads) < 1e-12] = other
            return Variable(
                pargs=self.pargs * other,
                grads=grads,
                identity=self.identity,
                origin_shape=self.origin_shape,
            )
        raise TypeError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1.0)

    def __truediv__(self, other):
        if other == 0:
            raise ValueError
        return self.__mul__(1.0 / other)

    def __rtruediv__(self, other):
        return self.__pow__(-1.0).__mul__(other)

    def __pow__(self, other):
        if isinstance(other, (int, float, np.float64)):
            grad = other * self.pargs ** (other - 1.0)
            if isinstance(self.pargs, np.float64):
                grads = grad if abs(self.grads) < 1e-12 else self.grads * grad
            else:
                grads = self.grads * grad
                grads[abs(grads) < 1e-12] = grad
            return Variable(
                pargs=self.pargs ** other,
                grads=grads,
                identity=self.identity,
                origin_shape=self.origin_shape,
            )
        raise TypeError

    def __str__(self):
        return "Variable(pargs={}, grads={})".format(self.pargs, self.grads)

    def copy(self):
        return Variable(
            pargs=self.pargs,
            grads=self.grads,
            identity=self.identity,
            origin_shape=self.origin_shape,
        )

    def zero_grad(self):
        self._grads = np.zeros(self._shape, dtype=np.float64)

    def flatten(self):
        return Variable(
            pargs=self._pargs.flatten(),
            grads=self._grads.flatten(),
            identity=self.identity,
            origin_shape=self._pargs.shape,
        )
