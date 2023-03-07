from __future__ import annotations

import copy
from typing import Union
import numpy as np

from QuICT.core.utils import (
    GateType, MatrixType, SPECIAL_GATE_SET, DIAGONAL_GATE_SET, CGATE_LIST,
    PAULI_GATE_SET, CLIFFORD_GATE_SET,
    perm_decomposition, matrix_product_to_circuit
)
from QuICT.tools.exception.core import (
    TypeError, ValueError, GateAppendError, GateQubitAssignedError,
    QASMError, GateMatrixError, GateParametersAssignedError
)

from .utils import GateMatrixGenerator, ComplexGateBuilder, InverseGate


class BasicGate(object):
    """ the abstract SuperClass of all basic quantum gate

    All basic quantum gate described in the framework have
    some common attributes and some common functions
    which defined in this class

    Attributes:
        controls(int): the number of the control bits of the gate
        cargs(list<int>): the list of the index of control bits in the circuit
        carg(int, read only): the first object of cargs

        targets(int): the number of the target bits of the gate
        targs(list<int>): the list of the index of target bits in the circuit
        targ(int, read only): the first object of targs

        params(list): the number of the parameter of the gate
        pargs(list): the list of the parameter
        parg(read only): the first object of pargs

        qasm_name(str, read only): gate's name in the OpenQASM 2.0
        type(GateType, read only): gate's type described by GateType

        matrix(np.array): the unitary matrix of the quantum gate act on targets
    """
    ####################################################################
    ############          Quantum Gate's property           ############
    ####################################################################
    @property
    def type(self) -> GateType:
        return self._type

    @property
    def matrix_type(self) -> MatrixType:
        return self._matrix_type

    @property
    def precision(self):
        return self._precision

    @property
    def qasm_name(self):
        return self._qasm_name
    
    @property
    def matrix(self) -> np.ndarray:
        # TODO: get from GateMatrix
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

    @property
    def target_matrix(self) -> np.ndarray:
        # TODO: get from GateMatrix
        return self.matrix

    ################    Quantum Gate's Target Qubits    ################
    @property
    def targets(self) -> int:
        return self._targets

    @property
    def targ(self):
        return self.targs[0]

    @property
    def targs(self):
        return self._targs

    @targs.setter
    def targs(self, targs: list):
        if isinstance(targs, int):
            targs = [targs]

        assert len(targs) == len(set(targs)), ValueError("BasicGate.targs", "not have duplicated value", targs)
        assert not set(self._cargs) & set(targs), ValueError(
            "BasicGate.targs", "have no same index with control qubits", set(self._cargs) & set(targs)
        )
        self._targs = targs

    ################    Quantum Gate's Control Qubits   ################
    @property
    def controls(self) -> int:
        return self._controls

    @property
    def carg(self):
        assert self._controls > 0, f"There is no control qubits for this gate, {self.type}"
        return self.cargs[0]

    @property
    def cargs(self):
        assert self._controls > 0, f"There is no control qubits for this gate, {self.type}"
        return self._cargs

    @cargs.setter
    def cargs(self, cargs: Union[list, int]):
        if isinstance(cargs, int):
            cargs = [cargs]

        assert len(cargs) == len(set(cargs)), ValueError("BasicGate.cargs", "not have duplicated value", cargs)
        self._cargs = cargs

    #################    Quantum Gate's parameters    ##################
    @property
    def params(self) -> int:
        return self._params

    @property
    def parg(self):
        assert self._params > 0, f"There is no parameters for this gate, {self.type}"
        return self.pargs[0]

    @property
    def pargs(self):
        assert self._params > 0, f"There is no parameters for this gate, {self.type}"
        return self._pargs

    @pargs.setter
    def pargs(self, pargs: list):
        if isinstance(pargs, list):
            self._pargs = pargs
        else:
            self._pargs = [pargs]

        if len(self._pargs) != self.params:
            raise ValueError("BasicGate.pargs:length", f"equal to gate's parameter number {self._pargs}", len(pargs))

    def __init__(
        self,
        controls: int,
        targets: int,
        params: int,
        type_: GateType,
        matrix_type: MatrixType = MatrixType.normal,
        pargs: list = [],
        precision: str = "double"
    ):
        assert isinstance(controls, int), TypeError("BasicGate.controls", "int", type(controls))
        assert isinstance(targets, int), TypeError("BasicGate.targets", "int", type(targets))
        assert isinstance(params, int), TypeError("BasicGate.params", "int", type(params))
        self._targets = targets
        self._targs = []    # list of int

        self._controls = controls
        if self._controls > 0:
            self._cargs = []    # list of int

        self._params = params
        if self._params > 0:
            if len(pargs) > 0:
                self.permit_element(pargs)

            self._pargs = pargs    # list of float/..

        assert isinstance(type_, GateType), TypeError("BasicGate.type", "GateType", type(type_))
        assert isinstance(matrix_type, MatrixType), TypeError("BasicGate.matrixtype", "MatrixType", type(type_))
        self._type = type_
        self._matrix_type = matrix_type

        assert precision in ["double", "single"], \
            ValueError("BasicGate.precision", "not within [double, single]", precision)
        self._precision = np.complex128 if precision == "double" else np.complex64
        self._matrix = None
        self._qasm_name = str(type_.name)
        self.assigned_qubits = []   # list of qubits' id

    def __or__(self, targets):
        """deal the operator '|'

        Use the syntax "gate | circuit" or "gate | Composite Gate"
        to add the gate into the circuit or composite gate
        Some Examples are like this:

        X       | circuit
        CX      | circuit([0, 1])
        Measure | CompositeGate

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
                2) CompositeGate

        Raise:
            TypeError: the type of other is wrong
        """
        try:
            targets.append(self)
        except Exception as e:
            raise GateAppendError(f"Failure to append gate {self.name} to targets, due to {e}")

    def __and__(self, targets):
        """deal the operator '&'

        Use the syntax "gate & int" or "gate & list<int>" to set gate's attribute.
        Special uses when in composite gate's context.

        Some Examples are like this:
        X       & 1
        CX      & [0, 1]

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) int
                2) list<int>

        Raise:
            TypeError: the type of targets is wrong
        """
        _gate = self.copy()

        if isinstance(targets, int):
            assert _gate.is_single(), GateQubitAssignedError("The qubits number should equal to the quantum gate.")

            _gate.targs = [targets]
        elif isinstance(targets, list):
            if len(targets) != _gate.controls + _gate.targets:
                raise GateQubitAssignedError("The qubits number should equal to the quantum gate.")

            _gate.cargs = targets[:_gate.controls]
            _gate.targs = targets[_gate.controls:]
        else:
            raise TypeError("BasicGate.&", "int or list<int>", type(targets))

        if CGATE_LIST:
            CGATE_LIST[-1].append(_gate)
        else:
            return _gate

    def __call__(self):
        """ give parameters for the gate, and give parameters by "()", and parameters should be one of int/float/complex

        Some Examples are like this:

        Rz(np.pi / 2)           | qubit
        U3(np.pi / 2, 0, 0)     | qubit

        *Important*: There is no parameters for current quantum gate.

        Returns:
            BasicGate: the gate after filled by parameters
        """
        return self.copy()

    def __eq__(self, other):
        assert isinstance(other, BasicGate), TypeError("BasicGate.==", "BasicGate", type(other))
        if (
            self.type != other.type or
            (self.cargs + self.targs) != (other.cargs + other.targs) or
            self.pargs != other.pargs
        ):
            return False

        if self.type == GateType.unitary and not np.allclose(self.matrix, other.matrix):
            return False

        return True

    def __str__(self):
        gstr = f"gate type: {self.type}; "
        if self.params > 0:
            gstr += f"parameters number: {self.params}; parameters: {self.pargs}; "

        if self.controls > 0:
            gstr += f"controls qubits' number: {self.controls}; controls qubits' indexes: {self.cargs}; "

        gstr += f"target qubits' number: {self.targets}; target qubits' indexes: {self.targs}"

        return gstr

    def __dict__(self):
        """ get gate information """
        gate_info = {
            "type": self.type,
            "parameters": self.pargs,
            "controls": self.controls,
            "control_bit": self.cargs,
            "targets": self.targets,
            "target_bit": self.targs
        }

        return gate_info

    def qasm(self):
        """ generator OpenQASM string for the gate

        Return:
            string: the OpenQASM 2.0 describe of the gate
        """
        if self.type in SPECIAL_GATE_SET[4:]:
            raise QASMError(f"The gate do not support qasm, {self.type}")

        qasm_string = self.qasm_name
        if self.params > 0:
            params = [str(parg) for parg in self.pargs]
            params_string = "(" + ", ".join(params) + ")"

            qasm_string += params_string

        ctargs = [f"q[{ctarg}]" for ctarg in self.cargs + self.targs]
        ctargs_string = " " + ', '.join(ctargs) + ";\n"
        qasm_string += ctargs_string

        return qasm_string

    def inverse(self):
        """ the inverse of the gate

        Return:
            BasicGate: the inverse of the gate
        """
        if self.params > 0:
            inverse_gargs = InverseGate.get_inverse_gate(self.type, self.pargs)
        else:
            inverse_gargs = InverseGate.get_inverse_gate(self.type)

        # Deal with inverse_gargs
        if inverse_gargs is None:
            return self

        if isinstance(inverse_gargs, tuple):
            gate_type, parameters = inverse_gargs
        else:
            gate_type = inverse_gargs

        # TODO: return new gate with parameters
        return None

    def commutative(self, goal, eps=1e-7):
        """ decide whether gate is commutative with another gate

        note when the gate is special gates like Unitary, Permutation, Measure and so on, return False.

        Args:
            goal(BasicGate): the target gate
            eps(float): the precision of comparision

        Return:
            bool: True if commutative
        """
        # Check the affected qubits
        self_controls = set(self.cargs)
        self_targets = set(self.targs)
        goal_controls = set(goal.cargs)
        goal_targets = set(goal.targs)
        # If the affected qubits of the gates are completely different, they must commute
        self_qubits = self_controls | self_targets
        goal_qubits = goal_controls | goal_targets
        if len(self_qubits & goal_qubits) == 0:
            return True

        # Ignore all the special gates except for the unitary gates due to the difficulty of getting the matrices
        if (
            (self.is_special() and self.type != GateType.unitary) or
            (goal.is_special() and self.type != GateType.unitary)
        ):
            return False

        # It means commuting that any of the target matrices is close to identity
        if (self.is_identity() or goal.is_identity()):
            return True

        # Check the target matrices of the gates
        A = self.target_matrix
        B = goal.target_matrix
        # For gates whose number of target qubits is 1, optimized judgment could be used
        if self.targets == 1 and goal.targets == 1:
            # Diagonal target gates commutes with the control qubits
            if (
                (len(self_controls & goal_targets) > 0 and not goal.is_diagonal()) or
                (len(goal_controls & self_targets) > 0 and not self.is_diagonal())
            ):
                return False
            # Compute the target matrix commutation
            if (
                len(goal_targets & self_targets) > 0 and
                not np.allclose(A.dot(B), B.dot(A), rtol=eps, atol=eps)
            ):
                return False

            return True
        # Otherwise, we need to calculate the matrix commutation directly
        else:
            # Collect all the affected qubits and create the converter to the minimal qubits
            qubits = self_qubits | goal_qubits
            qubits_dict = {}
            for idx, x in enumerate(qubits):
                qubits_dict[x] = idx
            # Create two masked gates whose affected qubits are the minimal ones
            self_masked = self.cargs + self.targs
            for i in range(len(self_masked)):
                self_masked[i] = qubits_dict[self_masked[i]]
            goal_masked = goal.cargs + goal.targs
            for i in range(len(goal_masked)):
                goal_masked[i] = qubits_dict[goal_masked[i]]
            # Compute the matrix commutation
            self_matrix = self.expand(list(qubits))
            goal_matrix = goal.expand(list(qubits))
            return np.allclose(self_matrix.dot(goal_matrix), goal_matrix.dot(self_matrix), rtol=eps, atol=eps)

    def is_single(self) -> bool:
        """ judge whether gate is a one qubit gate(excluding special gate like measure, reset, custom and so on)

        Returns:
            bool: True if it is a one qubit gate
        """
        return self.targets + self.controls == 1

    def is_control_single(self) -> bool:
        """ judge whether gate has one control bit and one target bit

        Returns:
            bool: True if it is has one control bit and one target bit
        """
        return self.controls == 1 and self.targets == 1

    def is_clifford(self) -> bool:
        """ judge whether gate's matrix is a Clifford gate

        Returns:
            bool: True if gate's matrix is a Clifford gate
        """
        return self.type in CLIFFORD_GATE_SET

    def is_diagonal(self) -> bool:
        """ judge whether gate's matrix is diagonal

        Returns:
            bool: True if gate's matrix is diagonal
        """
        return (self.type in DIAGONAL_GATE_SET or self.matrix_type == MatrixType.diagonal)

    def is_pauli(self) -> bool:
        """ judge whether gate's matrix is a Pauli gate

        Returns:
            bool: True if gate's matrix is a Pauli gate
        """
        return self.type in PAULI_GATE_SET

    def is_special(self) -> bool:
        """ judge whether gate's is special gate, which is one of
        [Measure, Reset, Barrier, Perm, Unitary, ...]

        Returns:
            bool: True if gate's matrix is special
        """
        return self.type in SPECIAL_GATE_SET

    def is_identity(self) -> bool:
        return self.matrix_type == MatrixType.identity 

    # TODO: keep or remove? weird
    def expand(self, qubits: Union[int, list]) -> bool:
        """ expand self matrix into the circuit's unitary linear space. If input qubits is integer, please make sure
        the indexes of current gate is within [0, qubits).

        Args:
            qubits Union[int, list]: the total number of qubits of the target circuit or the indexes of expand qubits.
        """
        if isinstance(qubits, int):
            qubits = list(range(qubits))

        qubits_num = len(qubits)
        if qubits_num == self.controls + self.targets:
            return self.matrix

        assert qubits_num > self.controls + self.targets, GateQubitAssignedError(
            "The expand qubits' num should >= gate's qubits num."
        )
        gate_args = self.cargs + self.targs
        if len(gate_args) == 0:     # Deal with not assigned quantum gate
            gate_args = [qubits[i] for i in range(self.controls + self.targets)]

        updated_args = [qubits.index(garg) for garg in gate_args]
        return matrix_product_to_circuit(self.matrix, updated_args, qubits_num)

    # TODO: try to refactoring this
    def copy(self):
        """ return a copy of this gate

        Returns:
            gate(BasicGate): a copy of this gate
        """
        class_name = str(self.__class__.__name__)
        gate = globals()[class_name]()

        if gate.type in SPECIAL_GATE_SET:
            gate.controls = self.controls
            gate.targets = self.targets
            gate.params = self.params

        if gate.type in [GateType.perm, GateType.unitary, GateType.perm_fx]:
            gate.matrix = copy.deepcopy(self.matrix)

        if self.params > 0:
            gate.pargs = self.pargs[:]

        gate.targs = self.targs[:]
        if self.controls > 0:
            gate.cargs = self.cargs[:]

        if self.assigned_qubits:
            gate.assigned_qubits = self.assigned_qubits[:]

        if self.precision == np.complex64:
            gate.convert_precision()

        return gate

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

        for el in element
            if not isinstance(el, (int, float, complex, np.complex64)):
                raise TypeError("basicGate.targs", "int/float/complex", type(el))


# TODO: using as copy
def gate_builder():
    pass
