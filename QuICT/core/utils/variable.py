import numpy as np
from typing import Union
import uuid

from QuICT.tools.exception.core import TypeError


class Variable(object):
    """Used to represent trainable parameters of parameterized quantum gates or parameterized quantum circuits.

    Args:
        pargs (Union[float, np.float64, np.ndarray]): Initial parameter values.
        grads (Union[float, np.float64, np.ndarray], optional): Initial parameter gradients. Defaults to None.
        identity (str, optional): The unique identity of the variable. Defaults to None.
        origin_shape (tuple, optional): The original shape of parameters. Defaults to None.

    Examples:
        >>> import numpy as np
        >>> from QuICT.core.circuit import Circuit
        >>> from QuICT.core.gate import *
        >>> var = Variable(np.array([2.0, -0.2]))
        >>> cir = Circuit(3)
        >>> H | cir
        >>> Rx(-var[0]) | cir(0)
        >>> Rx(var[1]) | cir(2)
        >>> Rzz(0.6 * var[0] ** 2 - 0.7) | cir([0, 1])
        >>> cir.draw("command")
                ┌───┐ ┌────────┐
        q_0: |0>┤ h ├─┤ rx(-2) ├──■────────
                ├───┤ └────────┘  │ZZ(1.7)
        q_1: |0>┤ h ├─────────────■────────
                ├───┤┌──────────┐
        q_2: |0>┤ h ├┤ rx(-0.2) ├──────────
                └───┘└──────────┘
    """

    @property
    def item(self) -> np.float64:
        """Get the item.

        Returns:
            np.float64: The item of the variable.
        """
        assert self._pargs.shape == (1,), "Can only get item from 1x1 matrix."
        return self._pargs[0]

    @property
    def pargs(self) -> np.ndarray:
        """Get the parameter values.

        Returns:
            np.ndarray: The parameter values of the variable.
        """
        return self._pargs

    @property
    def grads(self) -> np.ndarray:
        """Get the gradients.

        Returns:
            np.ndarray: The gradients of the variable.
        """
        return self._grads

    @property
    def identity(self) -> str:
        """Get the unique identity.

        Returns:
            str: The unique identity of the variable.
        """
        return self._identity

    @property
    def index(self) -> tuple:
        """Get the index.

        Returns:
            tuple: The index of the variable.
        """
        index = self.identity[32:]
        return tuple(map(int, index.split(", ")[:-1]))

    @property
    def shape(self) -> tuple:
        """Get the current shape.

        Returns:
            tuple: The current shape of the variable.
        """
        return self._shape

    @property
    def origin_shape(self) -> tuple:
        """Get the original shape when initialized.

        Returns:
            tuple: The original shape of the variable.
        """
        return self._origin_shape

    @pargs.setter
    def pargs(self, pargs):
        """Set the parameter values.

        Args:
            pargs (Union[float, np.float64, np.ndarray]): The parameter values.
        """
        if isinstance(pargs, np.ndarray):
            self._pargs = pargs.astype(np.float64)
        else:
            self._pargs = np.float64(pargs)

    @grads.setter
    def grads(self, grads):
        """Set the gradients.

        Args:
            grads (Union[float, np.float64, np.ndarray]): The gradients.
        """
        if isinstance(grads, np.ndarray):
            assert grads.shape == self._pargs.shape
            self._grads = grads.astype(np.float64)
        else:
            self._grads = np.float64(grads)

    @identity.setter
    def identity(self, uid):
        """Set the unique identity.

        Args:
            uid (str): The unique identity.
        """
        if isinstance(uid, str):
            self._identity = uid

    def __init__(
        self,
        pargs: Union[float, np.float64, np.ndarray],
        grads: Union[float, np.float64, np.ndarray] = None,
        identity: str = None,
        origin_shape: tuple = None,
    ):
        """Initialize a Variable instance."""
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
        """Get data according to index.

        Args:
            index: The index.

        Returns:
            Variable: self[index].
        """
        return Variable(
            pargs=self.pargs[index],
            grads=self.grads[index],
            identity=self.identity
            + str(index).replace("(", "").replace(")", "")
            + ", ",
            origin_shape=self.origin_shape,
        )

    def __eq__(self, other):
        """Check whether two variables are equal.

        Args:
            other (Variable): Another variable.

        Returns:
            bool: Whether two variables are equal.
        """
        if (self.pargs == other.pargs).all() and (self.grads == other.grads).all():
            return True
        else:
            return False

    def __add__(self, other):
        """Add a number or an array.

        Args:
            other (Union[int, float, np.float64, np.ndarray]): The number or array.

        Returns:
            Variable: The result of add.
        """
        assert isinstance(other, (int, float, np.float64, np.ndarray)), TypeError(
            "Variable.__add__.other", "int/float/np.float64/np.ndarray", type(other)
        )
        if isinstance(other, np.ndarray):
            assert (
                other.shape == self.pargs.shape
            ), "Shape mismatch in variables addition."
        return Variable(
            pargs=self.pargs + other,
            grads=self.grads,
            identity=self.identity,
            origin_shape=self.origin_shape,
        )

    def __radd__(self, other):
        """Add a number or an array."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract a number or an array"""
        return self.__add__(-other)

    def __rsub__(self, other):
        """Self subtract a number or an array"""
        return self.__mul__(-1.0).__add__(other)

    def __mul__(self, other):
        """Multiply a number."""
        assert isinstance(other, (int, float, np.float64)), TypeError(
            "Variable.__mul__.other", "int/float/np.float64", type(other)
        )
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

    def __rmul__(self, other):
        """Multiply a number."""
        return self.__mul__(other)

    def __neg__(self):
        """Return the negative of the variable."""
        return self.__mul__(-1.0)

    def __truediv__(self, other):
        """Divide a number."""
        if other == 0:
            raise ValueError
        return self.__mul__(1.0 / other)

    def __rtruediv__(self, other):
        """Divided by a number."""
        return self.__pow__(-1.0).__mul__(other)

    def __pow__(self, other):
        """Exponentiation operation."""
        assert isinstance(other, (int, float, np.float64)), TypeError(
            "Variable.__pow__.other", "int/float/np.float64", type(other)
        )
        grad = other * self.pargs ** (other - 1.0)
        if isinstance(self.pargs, np.float64):
            grads = grad if abs(self.grads) < 1e-12 else self.grads * grad
        else:
            grads = self.grads * grad
            grads[abs(grads) < 1e-12] = grad
        return Variable(
            pargs=self.pargs**other,
            grads=grads,
            identity=self.identity,
            origin_shape=self.origin_shape,
        )

    def __str__(self):
        return "Variable(pargs={}, grads={})".format(self.pargs, self.grads)

    def copy(self):
        """Return a copy of the variable."""
        return Variable(
            pargs=self.pargs,
            grads=self.grads,
            identity=self.identity,
            origin_shape=self.origin_shape,
        )

    def zero_grad(self):
        """Set gradients to 0."""
        self._grads = np.zeros(self._shape, dtype=np.float64)

    def flatten(self):
        """Return a copy of the variable whose pargs and grads are collapsed into one dimension."""
        return Variable(
            pargs=self._pargs.flatten(),
            grads=self._grads.flatten(),
            identity=self.identity,
            origin_shape=self._pargs.shape,
        )
