import numpy as np

from QuICT.core.gate import BasicGate
from QuICT.core.utils import GateType, MatrixType


class SymbolicBasicGate(BasicGate):
    def __init__(self,
                 controls: int,
                 targets: int,
                 params: int,
                 type_: GateType,
                 matrix_type: MatrixType = MatrixType.normal,
                 pargs: list = None,
                 precision: str = "double",
                 is_original_gate: bool = False):

        if pargs is None:
            pargs = []
        super().__init__(controls, targets, params, type_, matrix_type, pargs, precision, is_original_gate)

    def permit_element(self, element):
        """ judge whether the type of a parameter is int/float/complex

        for a quantum gate, the parameter should be int/float/complex

        Args:
            element: the element to be judged

        Returns:
            bool: True if the type of element is int/float/complex
        """
        if not isinstance(element, list):
            element = [element]

        for el in element:
            if not isinstance(el, (int, float, complex, np.complex64, SymbolicPhase)):
                raise TypeError("basicGate.targs", "int/float/complex/SymbolicPhase", type(el))


SymbolicRz = SymbolicBasicGate(0, 1, 1, GateType.rz, MatrixType.diagonal, is_original_gate=True)


class SymbolicPhaseVariable:
    """
    Symbolic phase variable.
    """
    __slots__ = ['phase', 'label']

    def __init__(self, label):
        self.phase = None
        self.label = label

    def __rmul__(self, other):
        ret = SymbolicPhase()
        if isinstance(other, float) or isinstance(other, int):
            ret.var_dict[self.label] = [self, other]
        else:
            raise TypeError('Left operator must be number')
        return ret

    def __truediv__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return (1 / other) * self
        else:
            raise TypeError('Left operator must be number')

    def __neg__(self):
        return -1 * self


class SymbolicPhase:
    """
    Symbolic phase expression.
    """
    __slots__ = ['var_dict', 'const']

    def __init__(self):
        self.var_dict = {}
        self.const = 0

    def copy(self):
        """
        Get a copy of this SymbolicPhase.

        Returns:
            SymbolicPhase: a copy
        """
        ret = SymbolicPhase()
        for k, v in self.var_dict.items():
            ret.var_dict[k] = v.copy()
        ret.const = self.const
        return ret

    def evaluate(self):
        """
        Evaluate the value of this SymbolicPhase.
        If it contains any undetermined variable, it will return float('inf').

        Returns:
            float: value of this SymbolicPhase
        """
        ret = self.const
        for var, coef in self.var_dict.values():
            if np.isclose(coef, 0):
                continue
            if var.phase is None:
                return float('inf')
            ret += var.phase * coef
        return ret

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        ret = self.copy()
        if isinstance(other, SymbolicPhase):
            ret.const += other.const
            for key, pack in other.var_dict.items():
                var, val = pack
                if key not in ret.var_dict:
                    ret.var_dict[key] = [var, 0]
                ret.var_dict[key][1] += val
                if np.isclose(ret.var_dict[key][1], 0):
                    ret.var_dict.pop(key)
        elif isinstance(other, SymbolicPhaseVariable):
            if other.label not in ret.var_dict:
                ret.var_dict[other.label] = [other, 0]
            ret.var_dict[other.label][1] += 1
            if np.isclose(ret.var_dict[other.label][1], 0):
                ret.var_dict.pop(other)
        elif isinstance(other, int) or isinstance(other, float):
            ret.const += other
        else:
            raise TypeError("Invalid right operator.")
        return ret

    def __rmul__(self, other):
        ret = self.copy()
        if isinstance(other, float) or isinstance(other, int):
            ret.const *= other
            for each in ret.var_dict:
                ret.var_dict[each][1] *= other
        else:
            raise TypeError('Left operator must be number')
        return ret

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __float__(self):
        return self.evaluate()

    def __mod__(self, other):
        ret = self.copy()
        if isinstance(other, float) or isinstance(other, int):
            ret.const = np.mod(ret.const, other)
        else:
            raise TypeError('right operator mush be number')
        return ret
