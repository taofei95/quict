from __future__ import annotations

from typing import Union
import numpy as np

from QuICT.core.utils import (
    Variable, matrix_product_to_circuit, GateType, MatrixType,
    CGATE_LIST, GATE_ARGS_MAP, PAULI_GATE_SET, CLIFFORD_GATE_SET, GATEINFO_MAP
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
        type(GateType, read only): gate's type described by GateType
        matrix_type(MatrixType, read only): gate matrix's type described by MatrixType
        precision(str): The gate's precision, one of [double, single]
        qasm_name(str, read only): gate's name in the OpenQASM 2.0
        matrix(np.array): the unitary matrix of the quantum gate act on qubits
        target_matrix(np.array): the unitary matrix of the quantum gate act on targets

        targets(int): the number of the target bits of the gate
        targs(list<int>): the list of the index of target bits in the circuit
        targ(int, read only): the first object of targs

        controls(int): the number of the control bits of the gate
        cargs(list<int>): the list of the index of control bits in the circuit
        carg(int, read only): the first object of cargs

        params(list): the number of the parameter of the gate
        pargs(list): the list of the parameter
        parg(read only): the first object of pargs
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

    @precision.setter
    def precision(self, precision: str):
        assert precision in ["double", "single"], \
            ValueError("BasicGate.precision", "not within [double, single]", precision)

        if precision != self._precision:
            self._precision = precision
            self._is_matrix_update = True

    @property
    def qasm_name(self):
        return self._qasm_name

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None or self._is_matrix_update:
            self._matrix = GateMatrixGenerator().get_matrix(self)
            self._is_matrix_update = False

        return self._matrix

    def get_matrix(self, precision) -> np.ndarray:
        return GateMatrixGenerator().get_matrix(self, precision)

    @property
    def target_matrix(self) -> np.ndarray:
        if self._target_matrix is None or self._is_matrix_update:
            self._target_matrix = GateMatrixGenerator().get_matrix(self, is_get_target=True)
            self._is_matrix_update = False

        return self._target_matrix

    @property
    def grad_matrix(self):
        if self._grad_matrix is None or self._is_matrix_update:
            self._grad_matrix = GateMatrixGenerator().get_matrix(self, is_get_grad=True)
            self._is_matrix_update = False

        return self._grad_matrix

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
        # assert self._controls > 0, f"There is no control qubits for this gate, {self.type}"
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
    def variables(self) -> int:
        return self._variables

    @property
    def parg(self):
        # assert self._params > 0, f"There is no parameters for this gate, {self.type}"
        if self._params == 0:
            return None

        return self.pargs[0]

    @property
    def pargs(self):
        # assert self._params > 0, f"There is no parameters for this gate, {self.type}"
        if self._params == 0:
            return []

        return self._pargs

    @pargs.setter
    def pargs(self, pargs: list):
        if isinstance(pargs, int):
            pargs = [pargs]

        self.permit_element(pargs)
        assert len(self._pargs) == self.params, \
            ValueError("BasicGate.pargs:length", f"equal to gate's parameter number {self._pargs}", len(pargs))

        self._pargs = pargs
        self._is_matrix_update = True

    def __init__(
        self,
        controls: int,
        targets: int,
        params: int,
        type_: GateType,
        matrix_type: MatrixType = MatrixType.normal,
        pargs: list = [],
        precision: str = "double",
        is_original_gate: bool = False
    ):
        """
        Args:
            controls (int): The number of control qubits
            targets (int): The number of target qubits
            params (int): The number of gate's parameters
            type_ (GateType): The gate's type
            matrix_type (MatrixType, optional): The gate matrix's type. Defaults to MatrixType.normal.
            pargs (list, optional): The gate's parameters. Defaults to [].
            precision (str, optional): The gate's precison, one of [double, single]. Defaults to "double".
            is_original_gate (bool, optional): Whether is the initial quantum gate, such as H. Defaults to False.
        """
        assert isinstance(controls, int), TypeError("BasicGate.controls", "int", type(controls))
        assert isinstance(targets, int), TypeError("BasicGate.targets", "int", type(targets))
        assert isinstance(params, int), TypeError("BasicGate.params", "int", type(params))
        self._targets = targets
        self._targs = []    # list of int

        self._controls = controls
        self._cargs = []    # list of int

        self._variables = 0
        self._params = params
        self._pargs = []
        if self._params > 0:
            if len(pargs) > 0:
                self.permit_element(pargs)
            else:
                pargs = GATE_ARGS_MAP[type_]
            self._pargs = pargs    # list of float/..

        assert isinstance(type_, GateType), TypeError("BasicGate.type", "GateType", type(type_))
        assert isinstance(matrix_type, MatrixType), TypeError("BasicGate.matrixtype", "MatrixType", type(type_))
        self._type = type_
        self._matrix_type = matrix_type

        assert precision in ["double", "single"], \
            ValueError("BasicGate.precision", "not within [double, single]", precision)
        self._precision = precision
        self._matrix = None
        self._target_matrix = None
        self._grad_matrix = None
        self._qasm_name = str(type_.name)
        self._is_matrix_update = False
        self._is_original = is_original_gate

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
            raise GateAppendError(f"Failure to append gate {self} to targets, due to {e}")

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
        if isinstance(targets, int):
            targets = [targets]

        if not isinstance(targets, list):
            raise TypeError("BasicGate.&", "int or list<int>", type(targets))

        assert len(targets) == self.controls + self.targets, \
            GateQubitAssignedError("The qubits number should equal to the quantum gate.")

        if self._is_original:
            _gate = self.copy()
        else:
            _gate = self

        _gate.cargs = targets[:_gate.controls]
        _gate.targs = targets[_gate.controls:]

        if CGATE_LIST:
            CGATE_LIST[-1].append(_gate)

        return _gate

    def __call__(self, *args):
        """ give parameters for the gate, and give parameters by "()", and parameters should be one of int/float/complex

        Some Examples are like this:

        Rz(np.pi / 2)
        U3(np.pi / 2, 0, 0)

        *Important*: There is no parameters for some quantum gate.

        Returns:
            BasicGate: the gate after filled by parameters
        """
        assert len(args) == self.params, \
            GateParametersAssignedError("The number of given parameters not matched the quantum gate.")

        if self._is_original:
            _gate = self.copy()
        else:
            _gate = self

        _gate.pargs = list(args)
        return _gate

    def __eq__(self, other):
        assert isinstance(other, BasicGate), TypeError("BasicGate.==", "BasicGate", type(other))
        if (
            self.type != other.type or
            self.matrix_type != other.matrix_type or
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

    def qasm(self, targs: list = None):
        """ generator OpenQASM string for the gate

        Return:
            string: the OpenQASM 2.0 describe of the gate
        """
        if self.type in [GateType.perm, GateType.perm_fx]:
            raise QASMError(f"This gate do not support qasm, {self.type}")

        qasm_string = self.qasm_name
        if self.params > 0:
            params = []
            for parg in self.pargs:
                if isinstance(parg, Variable):
                    params.append(str(parg.pargs))
                else:
                    params.append(str(parg))
            params_string = "(" + ", ".join(params) + ")"

            qasm_string += params_string

        qubit_idxes = self.cargs + self.targs if targs is None else targs
        ctargs = [f"q[{ctarg}]" for ctarg in qubit_idxes]
        ctargs_string = " " + ', '.join(ctargs) + ";\n"
        qasm_string += ctargs_string

        return qasm_string

    def inverse(self):
        """ the inverse of the quantum gate, if there is no inverse gate, return itself.

        Return:
            BasicGate: the inverse of the gate
        """
        inverse_gargs, inverse_pargs = InverseGate.get_inverse_gate(self.type, self.pargs)

        # Deal with inverse_gargs
        if inverse_gargs is None:
            return self

        inverse_gate = gate_builder(inverse_gargs, params=inverse_pargs)
        gate_args = self.cargs + self.targs
        if len(gate_args) > 0:
            inverse_gate & gate_args

        return inverse_gate

    def build_gate(self, qidxes: list = None):
        """ Gate Decomposition, which divided the current gate with a set of small gates. """
        if self.type == GateType.cu3:
            cgate = ComplexGateBuilder.build_gate(self.type, self.parg, self.matrix)
        else:
            gate_list = ComplexGateBuilder.build_gate(self.type, self.parg)
            if gate_list is None:
                return gate_list

            cgate = self._cgate_generator_from_build_gate(gate_list)

        gate_args = self.cargs + self.targs if qidxes is None else qidxes
        if len(gate_args) > 0:
            cgate & gate_args

        return cgate

    def _cgate_generator_from_build_gate(self, cgate_list: list):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        for gate_type, qidxes, pargs in cgate_list:
            gate = gate_builder(gate_type, params=pargs)
            gate | cgate(qidxes)

        return cgate

    def commutative(self, goal: BasicGate, eps=1e-7):
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
        return self.matrix_type in [MatrixType.diagonal, MatrixType.control]

    def is_pauli(self) -> bool:
        """ judge whether gate's matrix is a Pauli gate

        Returns:
            bool: True if gate's matrix is a Pauli gate
        """
        return self.type in PAULI_GATE_SET

    def is_special(self) -> bool:
        """ judge whether gate's is special gate, which is one of
        [Measure, Reset, Barrier, Perm, ...]

        Returns:
            bool: True if gate's matrix is special
        """
        return self.matrix_type == MatrixType.special

    def is_identity(self) -> bool:
        """ judge whether gate's matrix is identity matrix

        Returns:
            bool: True if gate's matrix is identity
        """
        return self.type == GateType.id or self.matrix_type == MatrixType.identity

    def expand(self, qubits: Union[int, list], device: str = "CPU") -> bool:
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
        return matrix_product_to_circuit(self.matrix, updated_args, qubits_num, device)

    def copy(self):
        """ return a copy of this gate

        Returns:
            gate(BasicGate): a copy of this gate
        """
        pargs = [
            parg.copy() if isinstance(parg, Variable) else parg for parg in self.pargs
        ]
        gate = BasicGate(
            self.controls, self.targets, self.params, self.type,
            self.matrix_type, pargs, self.precision
        )

        if len(self.targs) > 0:
            gate.targs = self.targs[:]

        if self.controls > 0 and len(self.cargs) > 0:
            gate.cargs = self.cargs[:]

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

        for el in element:
            if not isinstance(el, (int, float, complex, np.complex64, Variable)):
                raise TypeError("basicGate.pargs", "int/float/complex/Variable", type(el))
            if isinstance(el, Variable):
                self._variables += 1
                if not isinstance(el.pargs, (int, float, complex, np.complex64)):
                    raise TypeError("basicGate.pargs", "int/float/complex/Variable", type(el.pargs))


class Unitary(BasicGate):
    """ The class about the Unitary Quantum Gate """
    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self.validate_matrix_shape(matrix)
        assert int(np.log2(matrix.shape[0])) == self.controls + self.targets, \
            "Only support redefined unitary matrix with same size as before"
        self._matrix = matrix

    def get_matrix(self, precision) -> np.ndarray:
        _dtype = np.complex128 if precision == "double" else np.complex64
        return self._matrix.astype(_dtype)

    @property
    def target_matrix(self):
        return self._matrix

    def __init__(self, matrix: Union[list, np.ndarray], matrix_type: MatrixType = None):
        """
        Args:
            matrix (Union[list, np.ndarray]): The unitary matrix.
            matrix_type (MatrixType, optional): The matrix's type. Defaults to None.
        """
        # Validate matrix type
        self.validate_matrix_shape(matrix)

        # Validate Matrix Type
        if matrix_type is None:
            matrix_type, controls = self.validate_matrix_type(matrix)
        else:
            matrix_type = matrix_type
            controls = 0

        n = int(np.log2(matrix.shape[0]))
        if controls == n:
            controls = 0

        precision = "double" if matrix.dtype == np.complex64 else "single"
        super().__init__(
            controls=controls, targets=(n - controls), params=0,
            type_=GateType.unitary, matrix_type=matrix_type, precision=precision
        )
        self._matrix = matrix

    def validate_matrix_shape(self, matrix: np.ndarray):
        if isinstance(matrix, list):
            matrix = np.array(matrix)

        # Validate precision
        assert np.issubdtype(matrix.dtype, np.complex128) or np.issubdtype(matrix.dtype, np.complex64), \
            TypeError("unitary.matrix.dtype", "complex64/128", matrix.dtype)

        # Validate matrix shape is square
        length, width = matrix.shape
        if length != width:
            N = int(np.log2(matrix.size))
            assert N ^ 2 == matrix.size, GateMatrixError("the shape of unitary matrix should be square.")
            matrix = matrix.reshape(N, N)

        n = int(np.log2(matrix.shape[0]))
        if (1 << n) != matrix.shape[0]:
            raise GateMatrixError("the length of list should be the square of power(2, n)")

    @staticmethod
    def validate_matrix_type(matrix: np.ndarray) -> MatrixType:
        """ Check the matrix's type about given unitary matrix

        Args:
            matrix (np.ndarry): The given unitary matrix

        Returns:
            MatrixType: The matrix type
        """
        length = matrix.shape[0]
        matrix_type = matrix.dtype

        controls = 0
        row_counter = 0
        while length // 2 > 0:
            id_length = length // 2
            if (
                np.allclose(
                    matrix[row_counter:(row_counter + id_length), row_counter:(row_counter + id_length)],
                    np.identity(id_length, dtype=matrix_type)
                ) and
                np.sum(matrix[row_counter:(row_counter + id_length), (row_counter + id_length):]) == 0 and
                np.sum(matrix[(row_counter + id_length):, row_counter:(row_counter + id_length)]) == 0
            ):
                controls += 1
                row_counter += id_length
                length //= 2
            else:
                break

        # Validate Identity Matrix or Control Matrix
        if length == 1:
            matrix_type = MatrixType.identity if matrix[-1, -1] == 1 else MatrixType.control
            return matrix_type, controls

        # Validate Diagonal Matrix or Reverse Matrix or Normal Matrix
        if np.allclose(matrix[row_counter:, row_counter:], np.diag(np.diag(matrix[row_counter:, row_counter:]))):
            matrix_type = MatrixType.diagonal
        else:
            reverse_mat = np.fliplr(matrix[row_counter:, row_counter:])
            if np.allclose(reverse_mat, np.diag(np.diag(reverse_mat))):
                matrix_type = MatrixType.reverse
            else:
                matrix_type = MatrixType.normal

        return matrix_type, controls

    def build_gate(self, qidxes: list = None):
        decomp_gate = ComplexGateBuilder.build_unitary(self._matrix)

        gate_args = self.cargs + self.targs if qidxes is None else qidxes
        if len(gate_args) > 0:
            decomp_gate & gate_args

        decomp_gate.gate_decomposition(decomposition=False)
        return decomp_gate

    def inverse(self):
        inverse_matrix = np.asmatrix(self.matrix).H

        gate_args = self.cargs + self.targs
        if len(gate_args) > 0:
            return Unitary(inverse_matrix) & gate_args

        return Unitary(inverse_matrix)

    def copy(self):
        _gate = Unitary(self.matrix, self.matrix_type)

        if len(self.targs) > 0:
            _gate.targs = self.targs[:]

        return _gate


class Perm(BasicGate):
    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = self._build_matrix()

        return self._matrix

    @property
    def target_matrix(self):
        return self.matrix

    def __init__(self, targets: int, params: list):
        """
        Args:
            n (int): the number of target qubits
            params (list[int]): the list of index, and the index represent which should be 1.

        Returns:
            PermFxGate: the gate after filled by parameters
        """
        if not isinstance(params, list) or not isinstance(targets, int):
            raise TypeError(f"targets must be int {type(targets)}, params must be list {type(params)}")

        assert len(params) == targets, GateParametersAssignedError("the length of params must equal to targets")
        assert len(set(params)) == targets, ValueError("PermGate.params", "have no duplicated value", params)

        pargs = []
        for parg in params:
            if not isinstance(parg, int):
                raise TypeError("PermGate.params.values", "int", type(parg))

            if parg < 0 or parg >= targets:
                raise ValueError("PermGate.params.values", f"[0, {targets}]", parg)

            pargs.append(parg)

        super().__init__(0, targets, targets, GateType.perm, MatrixType.normal, pargs)

    def _build_matrix(self):
        matrix_ = np.zeros((1 << self.targets, 1 << self.targets), dtype=self.precision)
        for idx, p in enumerate(self.pargs):
            if isinstance(p, Variable):
                matrix_[idx, p.pargs] = 1
            else:
                matrix_[idx, p] = 1

        return matrix_

    def inverse(self):
        inverse_targs = [self.targets - 1 - t for t in self.pargs]

        return Perm(self.targets, inverse_targs)


class PermFx(Perm):
    def __init__(self, targets: int, params: list):
        """
        Args:
            n (int): the number of target qubits
            params (list[int]): the list of index, and the index represent which should be 1.

        Returns:
            PermFxGate: the gate after filled by parameters
        """
        if not isinstance(params, list) or not isinstance(targets, int):
            raise TypeError(f"targets must be int {type(targets)}, params must be list {type(params)}")

        N = 1 << targets
        for p in params:
            assert p >= 0 and p < N, Exception("the params should be less than N")

        targets = targets + 1
        parameters = 1 << targets
        pargs = []
        for idx in range(1 << targets):
            if idx >> 1 in params:
                pargs.append(idx ^ 1)
            else:
                pargs.append(idx)

        super().__init__(0, targets, parameters, GateType.perm_fx, MatrixType.normal, pargs)


class MultiControlGate(BasicGate):
    def __init__(self, controls: int, gate_type: GateType, precision: str = "double", params: list = []):
        assert controls >= 0, ValueError("MultiControlGate.controls", ">= 0", controls)
        self._multi_controls = controls
        if gate_type not in GATEINFO_MAP.keys():
            raise TypeError("MultiControlGate.gate_type", "only support for QuICT Gate", gate_type)

        gate_info = list(GATEINFO_MAP[gate_type])
        gate_info[0] += controls
        super().__init__(*gate_info, params, precision)

    def inverse(self):
        """ the inverse of the quantum gate, if there is no inverse gate, return itself.

        Return:
            BasicGate: the inverse of the gate
        """
        inverse_gargs, inverse_pargs = InverseGate.get_inverse_gate(self.type, self.pargs)

        # Deal with inverse_gargs
        if inverse_gargs is None:
            return self

        inverse_gate = MultiControlGate(self._multi_controls, inverse_gargs, self.precision, params=inverse_pargs)
        gate_args = self.cargs + self.targs
        if len(gate_args) > 0:
            inverse_gate & gate_args

        return inverse_gate

    def build_gate(self):
        pass

    def copy(self):
        _gate = MultiControlGate(self._multi_controls, self.type, self.precision, self.pargs)
        gate_args = self.cargs + self.targs
        if len(gate_args) > 0:
            _gate & gate_args

        return _gate


def gate_builder(gate_type, precision: str = "double", params: list = [], random_params: bool = False) -> BasicGate:
    """ Build the target Quantum Gate.

    Args:
        gate_type (GateType): The gate's type.  \n
        precision (str, optional): The gate's precision. Defaults to "double".  \n
        params (list, optional): The gate's parameters. Defaults to [].  \n
        random_params (bool, optional): Whether using random parameters. Defaults to False.

    Returns:
        BasicGate: The class of target quantum gate
    """
    if gate_type not in GATEINFO_MAP.keys():
        raise TypeError("gate_builder.gate_type", "only support for fixed qubits gate", gate_type)

    gate_info = GATEINFO_MAP[gate_type]
    if random_params:
        params = list(np.random.uniform(0, 2 * np.pi, gate_info[2]))

    return BasicGate(
        *gate_info, params, precision
    )


H = BasicGate(*GATEINFO_MAP[GateType.h], is_original_gate=True)
r"""
Single-Qubit Hadamard Gate, which apply a pi rotation about the
X and Z axis.

[Matrix Representation]

$$
H = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & 1 \\
1 & -1 \\
\end{bmatrix}
$$

[How to apply]
```python
H | circuit(0)
H & 0 | circuit
```

[Graph Representation]
```python
      ┌───┐
q0: ──┤ H ├──
      └───┘
```

[QASM Representation]
```
h q[0]
```
"""

Hy = BasicGate(*GATEINFO_MAP[GateType.hy], is_original_gate=True)
r"""
Single-Qubit Hy Gate.

[Matrix Representation]

$$
Hy = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & -i \\
i & -1 \\
\end{bmatrix}
$$

[How to apply]
```python
Hy | circuit(0)
Hy & 0 | circuit
```

[Graph Representation]
```python
      ┌────┐
q0: ──┤ Hy ├──
      └────┘
```

[QASM Representation]
```
hy q[0]
```
"""

S = BasicGate(*GATEINFO_MAP[GateType.s], is_original_gate=True)
r"""
Single-Qubit S Gate, same as $Z^0.5$ Gate. It produce a phase of pi/2.

[Matrix Representation]

$$
S = \begin{bmatrix}
1 & 0 \\
0 & i \\
\end{bmatrix}
$$

[How to apply]
```python
S | circuit(0)
S & 0 | circuit
```

[Graph Representation]
```
      ┌───┐
q0: ──┤ S ├──
      └───┘
```

[QASM Representation]
```
s q[0]
```
"""

S_dagger = BasicGate(*GATEINFO_MAP[GateType.sdg], is_original_gate=True)
r"""
Single-Qubit S_dagger Gate, which produce a phase of -pi / 2.

[Matrix Representation]

$$
Sdg = \begin{bmatrix}
1 & 0 \\
0 & -i \\
\end{bmatrix}
$$

[How to apply]
```python
S_dagger | circuit(0)
S_dagger & 0 | circuit
```

[Graph Representation]
```
      ┌─────┐
q0: ──┤ Sdg ├──
      └─────┘
```

[QASM Representation]
```
sdg q[0]
```
"""

X = BasicGate(*GATEINFO_MAP[GateType.x], is_original_gate=True)
r"""
Single-Qubit Pauli-X Gate.

[Matrix Representation]

$$
X = \begin{bmatrix}
0 & 1 \\
1 & 0 \\
\end{bmatrix}
$$

[How to apply]
```python
X | circuit(0)
X & 0 | circuit
```

[Graph Representation]
```python
       ┌───┐
     ──┤ X ├──
       └───┘
```

[QASM Representation]
```
x q[0]
```
"""

Y = BasicGate(*GATEINFO_MAP[GateType.y], is_original_gate=True)
r"""
Single-Qubit Pauli-Y Gate.

[Matrix Representation]

$$
Y = \begin{bmatrix}
0 & -i \\
i & 0 \\
\end{bmatrix}
$$

[How to apply]
```python
Y | circuit(0)
Y & 0 | circuit
```

[Graph Representation]
```python
      ┌───┐
q0: ──┤ Y ├──
      └───┘
```

[QASM Representation]
```
y q[0]
```
"""

Z = BasicGate(*GATEINFO_MAP[GateType.z], is_original_gate=True)
r"""
Single-Qubit Pauli-Z Gate.

[Matrix Representation]

$$
Z = \begin{bmatrix}
1 & 0 \\
0 & -1 \\
\end{bmatrix}
$$

[How to apply]
```python
Z | circuit(0)
Z & 0 | circuit
```

[Graph Representation]
```python
      ┌───┐
q0: ──┤ Z ├──
      └───┘
```

[QASM Representation]
```
z q[0]
```
"""

SX = BasicGate(*GATEINFO_MAP[GateType.sx], is_original_gate=True)
r"""
Single-Qubit Sqrt-X Gate.

[Matrix Representation]

$$
SX = \frac{1}{2} \begin{bmatrix}
1 + i & 1 - i \\
1 - i & 1 + i \\
\end{bmatrix}
$$

[How to apply]
```python
SX | circuit(0)
SX & 0 | circuit
```

[Graph Representation]
```python
      ┌────┐
q0: ──┤ SX ├──
      └────┘
```

[QASM Representation]
```
sx q[0]
```
"""

SY = BasicGate(*GATEINFO_MAP[GateType.sy], is_original_gate=True)
r"""
Single-Qubit Sqrt-Y Gate.

[Matrix Representation]

$$
SY = \frac{1}{\sqrt{2}} \begin{bmatrix}
1 & -1 \\
1 & 1 \\
\end{bmatrix}
$$

[How to apply]
```python
SY | circuit(0)
SY & 0 | circuit
```

[Graph Representation]
```python
      ┌────┐
q0: ──┤ SY ├──
      └────┘
```

[QASM Representation]
```
sy q[0]
```
"""

SW = BasicGate(*GATEINFO_MAP[GateType.sw], is_original_gate=True)
r"""
Single-Qubit Sqrt-W Gate.

[Matrix Representation]

$$
SW = \begin{bmatrix}
\frac{1}{\sqrt{2}} & -\sqrt{\frac{i}{2}} \\
\sqrt{\frac{-i}{2}} & \frac{1}{\sqrt{2}} \\
\end{bmatrix}
$$

[How to apply]
```python
SW | circuit(0)
SW & 0 | circuit
```

[Graph Representation]
```python
      ┌────┐
q0: ──┤ SW ├──
      └────┘
```

[QASM Representation]
```
sw q[0]
```
"""

ID = BasicGate(*GATEINFO_MAP[GateType.id], is_original_gate=True)
r"""
Single-Qubit Identity Gate.

[Matrix Representation]

$$
ID = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
\end{bmatrix}
$$

[How to apply]
```python
ID | circuit(0)
ID & 0 | circuit
```

[Graph Representation]
```python
      ┌────┐
q0: ──┤ ID ├──
      └────┘
```

[QASM Representation]
```
id q[0]
```
"""

U1 = BasicGate(*GATEINFO_MAP[GateType.u1], is_original_gate=True)
r"""
Single-Qubit Rotation Gate, which apply an $\lambda$ rotation about Z axis.

[Matrix Representation]

$$
U1(\lambda) = \begin{bmatrix}
1 & 0 \\
0 & e^{i\lambda} \\
\end{bmatrix}
$$

[How to apply]
```python
U1(pi) | circuit(0)
U1(pi) & 0 | circuit
```

[Graph Representation]
```python
      ┌────────┐
q0: ──┤ U1(pi) ├──
      └────────┘
```

[QASM Representation]
```
u1(pi) q[0]
```
"""

U2 = BasicGate(*GATEINFO_MAP[GateType.u2], is_original_gate=True)
r"""
Single-Qubit Rotation Gate about X and Z axis.

[Matrix Representation]

$$
U2(\sigma, \lambda) = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & -e^{i\lambda} \\
e^{i\sigma} & e^{i(\lambda + \sigma)} \\
\end{bmatrix}
$$

[How to apply]
```python
U2(0, pi) | circuit(0)
U2(0, pi) & 0 | circuit
```

[Graph Representation]
```python
      ┌───────────┐
q0: ──┤ U2(0, pi) ├──
      └───────────┘
```

[QASM Representation]
```
u2(0, pi) q[0]
```
"""

U3 = BasicGate(*GATEINFO_MAP[GateType.u3], is_original_gate=True)
r"""
Single-Qubit Rotation Gate with three Euler Angles.

[Matrix Representation]

$$
U3(\theta, \sigma, \lambda) = \begin{bmatrix}
\cos(\frac{\theta}{2}) & -e^{i\lambda}\sin(\frac{\theta}{2}) \\
e^{i\sigma}\sin(\frac{\theta}{2}) & e^{i(\lambda + \sigma)}\cos(\frac{\theta}{2}) \\
\end{bmatrix}
$$

[How to apply]
```python
U3(0, pi, 1) | circuit(0)
U3(0, pi, 1) & 0 | circuit
```

[Graph Representation]
```python
      ┌──────────────┐
q0: ──┤ U3(0, pi, 1) ├──
      └──────────────┘
```

[QASM Representation]
```
u3(0, pi, 1) q[0]
```
"""

Rx = BasicGate(*GATEINFO_MAP[GateType.rx], is_original_gate=True)
r"""
Single-Qubit Rotation Gate, which apply an $\lambda$ rotation about X axis.

[Matrix Representation]

$$
Rx(\lambda) = \begin{bmatrix}
\cos(\frac{\lambda}{2}) & -i\sin(\frac{\lambda}{2}) \\
i\sin(\frac{\lambda}{2}) & \cos(\frac{\lambda}{2}) \\
\end{bmatrix}
$$

[How to apply]
```python
Rx(pi) | circuit(0)
Rx(pi) & 0 | circuit
```

[Graph Representation]
```python
      ┌────────┐
q0: ──┤ Rx(pi) ├──
      └────────┘
```

[QASM Representation]
```
rx(pi) q[0]
```
"""

Ry = BasicGate(*GATEINFO_MAP[GateType.ry], is_original_gate=True)
r"""
Single-Qubit Rotation Gate, which apply an $\lambda$ rotation about Y axis.
[Matrix Representation]

$$
Ry(\lambda) = \begin{bmatrix}
\cos(\frac{\lambda}{2}) & -\sin(\frac{\lambda}{2}) \\ 
\sin(\frac{\lambda}{2}) & \cos(\frac{\lambda}{2}) \\
\end{bmatrix}
$$

[How to apply]
```python
Ry(pi) | circuit(0)
Ry(pi) & 0 | circuit
```

[Graph Representation]
```python
      ┌────────┐
q0: ──┤ Ry(pi) ├──
      └────────┘
```

[QASM Representation]
```
ry(pi) q[0]
```
"""

Rz = BasicGate(*GATEINFO_MAP[GateType.rz], is_original_gate=True)
r"""
Single-Qubit Rotation Gate, which apply an $\lambda$ rotation about Z axis.

[Matrix Representation]

$$
Rz(\lambda) = \begin{bmatrix}
e^{\frac{-i\lambda}{2}} & 0 \\ 
0 & e^{\frac{i\lambda}{2}} \\
\end{bmatrix}
$$

[How to apply]
```python
Rz(pi) | circuit(0)
Rz(pi) & 0 | circuit
```

[Graph Representation]
```python
      ┌────────┐
q0: ──┤ Rz(pi) ├──
      └────────┘
```

[QASM Representation]
```
rz(pi) q[0]
```
"""

T = BasicGate(*GATEINFO_MAP[GateType.t], is_original_gate=True)
r"""
Single-Qubit T Gate, which produce a $\frac{pi}{4}$ phase. Its matrix equals $Rz(\frac{pi}{4})$.

[Matrix Representation]

$$
T = \begin{bmatrix}
1 & 0 \\ 
0 & \frac{1}{\sqrt{2}} + \frac{i}{\sqrt{2}} \\
\end{bmatrix}
$$

[How to apply]
```python
T | circuit(0)
T & 0 | circuit
```

[Graph Representation]
```python
      ┌───┐
q0: ──┤ T ├──
      └───┘
```

[QASM Representation]
```
t q[0]
```
"""

T_dagger = BasicGate(*GATEINFO_MAP[GateType.tdg], is_original_gate=True)
r"""
Single-Qubit T Dagger Gate, which produce a -pi/4 phase.

[Matrix Representation]

$$
Tdg = \begin{bmatrix}
1 & 0 \\ 
0 & \frac{1}{\sqrt{2}} - \frac{i}{\sqrt{2}} \\
\end{bmatrix}
$$

[How to apply]
```python
T_dagger | circuit(0)
T_dagger & 0 | circuit
```

[Graph Representation]
```python
      ┌─────┐
q0: ──┤ Tdg ├──
      └─────┘
```

[QASM Representation]
```
tdg q[0]
```
"""

Phase = BasicGate(*GATEINFO_MAP[GateType.phase], is_original_gate=True)
r"""
Single-Qubit Rotation Gate about Z axis.

[Matrix Representation]

$$
Phase(\lambda) = \begin{bmatrix}
1 & 0 \\
0 & e^{i\lambda} \\
\end{bmatrix}
$$

[How to apply]
```python
Phase(pi) | circuit(0)
Phase(pi) & 0 | circuit
```

[Graph Representation]
```python
      ┌───────────┐
q0: ──┤ phase(pi) ├──
      └───────────┘
```

[QASM Representation]
```
phase(pi) q[0]
```
"""

GPhase = BasicGate(*GATEINFO_MAP[GateType.gphase], is_original_gate=True)
r"""
Global Phase Gate.

[Matrix Representation]

$$
GPhase(\lambda) = \begin{bmatrix}
e^{i\lambda} & 0 \\ 
0 & e^{i\lambda} \\
\end{bmatrix}
$$

[How to apply]
```python
GPhase(pi) | circuit(0)
GPhase(pi) & 0 | circuit
```

[Graph Representation]
```python
      ┌────────────┐
q0: ──┤ gphase(pi) ├──
      └────────────┘
```

[QASM Representation]
```
gphase(pi) q[0]
```
"""

CZ = BasicGate(*GATEINFO_MAP[GateType.cz], is_original_gate=True)
r"""
Controlled Z Gate.

[Matrix Representation]

$$
CZ_{q0,q1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & -1 \\ 
\end{bmatrix}
$$

[How to apply]
```python
CZ | circuit([0, 1])
CZ & [0, 1] | circuit
```

[Graph Representation]
```python
    q0:  ────■────
             |
           ┌───┐
    q1:  ──┤ Z ├──
           └───┘
```

[QASM Representation]
```
CZ q[0], q[1]
```
"""

CX = BasicGate(*GATEINFO_MAP[GateType.cx], is_original_gate=True)
r"""
Controlled X Gate.

[Matrix Representation]

$$
CX_{q0,q1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 1 \\ 
0 & 0 & 1 & 0 \\ 
\end{bmatrix}
$$

[How to apply]
```python
CX | circuit([0, 1])
CX & [0, 1] | circuit
```

[Graph Representation]
```python
    q0:  ────■────
             |
           ┌───┐
    q1:  ──┤ Z ├──
           └───┘
```

[QASM Representation]
```
CX q[0], q[1]
```
"""

CY = BasicGate(*GATEINFO_MAP[GateType.cy], is_original_gate=True)
r"""
Controlled Y Gate.

[Matrix Representation]

$$
CY_{q0,q1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & i \\ 
0 & 0 & -i & 0 \\ 
\end{bmatrix}
$$

[How to apply]
```python
CY | circuit([0, 1])
CY & [0, 1] | circuit
```

[Graph Representation]
```python
    q0:  ────■────
             |
           ┌───┐
    q1:  ──┤ Y ├──
           └───┘
```

[QASM Representation]
```
CY q[0], q[1]
```
"""

CH = BasicGate(*GATEINFO_MAP[GateType.ch], is_original_gate=True)
r"""
Controlled H Gate.

[Matrix Representation]

$$
CH_{q0,q1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\
0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
0 & 0 & \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{2}} \\
\end{bmatrix}
$$

[How to apply]
```python
CH | circuit([0, 1])
CH & [0, 1] | circuit
```

[Graph Representation]
```python
    q0:  ────■────
             |
           ┌───┐
    q1:  ──┤ H ├──
           └───┘
```

[QASM Representation]
```
CH q[0], q[1]
```
"""

CRy = BasicGate(*GATEINFO_MAP[GateType.cry], is_original_gate=True)
r"""
Controlled Ry Gate.

[Matrix Representation]

$$
CRy_{q0,q1}(\lambda) = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 \\
0 & 0 & \cos(\frac{\lambda}{2}) & \sin(\frac{\lambda}{2}) \\
0 & 0 & \sin(\frac{\lambda}{2}) & \cos(\frac{\lambda}{2}) \\
\end{bmatrix}
$$

[How to apply]
```python
CRy(pi) | circuit([0, 1])
CRy(pi) & [0, 1] | circuit
```

[Graph Representation]
```python
    q0:  ──────■───────
               |
           ┌────────┐
    q1:  ──┤ Ry(pi) ├──
           └────────┘
```

[QASM Representation]
```
cry(pi) q[0], q[1]
```
"""

CRz = BasicGate(*GATEINFO_MAP[GateType.crz], is_original_gate=True)
r"""
Controlled Rz Gate.

[Matrix Representation]

$$
CRz_{q0,q1}(\lambda) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
1 & 0 & e^{\frac{-i\lambda}{2}} & 0 \\ 
1 & 0 & 0 & e^{\frac{i\lambda}{2}} \\ 
\end{bmatrix}
$$

[How to apply]
```python
CRz(pi) | circuit([0, 1])
CRz(pi) & [0, 1] | circuit
```

[Graph Representation]
```python
    q0:  ──────■───────
               |
           ┌────────┐
    q1:  ──┤ Rz(pi) ├──
           └────────┘
```

[QASM Representation]
```
crz(pi) q[0], q[1]
```
"""

CU1 = BasicGate(*GATEINFO_MAP[GateType.cu1], is_original_gate=True)
r"""
Controlled U1 Gate.

[Matrix Representation]

$$
CU1_{q0,q1}(\lambda) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & e^{i\lambda} \\
\end{bmatrix}
$$

[How to apply]
```python
CU1(pi) | circuit([0, 1])
CU1(pi) & [0, 1] | circuit
```

[Graph Representation]
```python
    q0:  ──────■───────
               |
           ┌────────┐
    q1:  ──┤ U1(pi) ├──
           └────────┘
```

[QASM Representation]
```
cu1(pi) q[0], q[1]
```
"""

CU3 = BasicGate(*GATEINFO_MAP[GateType.cu3], is_original_gate=True)
r"""
Controlled U3 Gate.

[Matrix Representation]

$$
CU3_{q0,q1}(\theta, \sigma, \lambda) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & \cos(\frac{\theta}{2}) & -\sin(\frac{\theta}{2}) e^{i\lambda} \\
0 & 0 & \sin(\frac{\theta}{2}) e^{i\sigma} & \cos(\frac{\theta}{2}) e^{i(\lambda + \sigma)} \\
\end{bmatrix}
$$

[How to apply]
```python
CU3(pi, 0, 1) | circuit([0, 1])
CU3(pi, 0, 1) & [0, 1] | circuit
```

[Graph Representation]
```python
    q0:  ─────────■──────────
                  |
           ┌──────────────┐
    q1:  ──┤ U3(pi, 0, 1) ├──
           └──────────────┘
```

[QASM Representation]
```
cu3(pi, 0, 1) q[0], q[1]
```
"""

FSim = BasicGate(*GATEINFO_MAP[GateType.fsim], is_original_gate=True)
r"""
Fermions Simulation Quantum Gate.

[Matrix Representation]

$$
FSim_{q0,q1}(\sigma, \lambda) = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & \cos(\theta) & -i\sin(\theta) & 0 \\
0 & -i\sin(\theta) & \cos(\theta) & 0 \\
0 & 0 & 0 & e^{-i\lambda} \\
\end{bmatrix}
$$

[How to apply]
```python
FSim(pi, 0) | circuit([0, 1])
FSim(pi, 0) & [0, 1] | circuit
```

[Graph Representation]
```python
           ┌──────────────┐
    q0:  ──┤0             ├──
           |  FSim(pi, 0) |
    q1:  ──┤1             ├──
           └──────────────┘
```

[QASM Representation]
```
fsim(pi, 0) q[0], q[1]
```
"""

Rxx = BasicGate(*GATEINFO_MAP[GateType.rxx], is_original_gate=True)
r"""
Double-Qubits X $\otimes$ X Gate.

[Matrix Representation]

$$
Rxx_{q0,q1}(\lambda) = \begin{bmatrix}
\cos(\frac{\lambda}{2}) & 0 & 0 & -i\sin(\frac{\lambda}{2}) \\  
0 & \cos(\frac{\lambda}{2}) & -i\sin(\frac{\lambda}{2}) & 0 \\
0 & -i\sin(\frac{\lambda}{2}) & \cos(\frac{\lambda}{2}) & 0 \\
-i\sin(\frac{\lambda}{2}) & 0 & 0 & \cos(\frac{\lambda}{2}) \\
\end{bmatrix}
$$

[How to apply]
```python
Rxx(pi) | circuit([0, 1])
Rxx(pi) & [0, 1] | circuit
```

[Graph Representation]
```python
           ┌───────────┐
    q0:  ──┤0          ├──
           |  Rxx(pi)  |
    q1:  ──┤1          ├──
           └───────────┘
```

[QASM Representation]
```
rxx(pi) q[0], q[1]
```
"""

Ryy = BasicGate(*GATEINFO_MAP[GateType.ryy], is_original_gate=True)
r"""
Double-Qubits Y $\otimes$ Y Gate.

[Matrix Representation]

$$
Ryy_{q0,q1}(\lambda) = \begin{bmatrix}
\cos(\frac{\lambda}{2}) & 0 & 0 & i\sin(\frac{\lambda}{2}) \\
0 & \cos(\frac{\lambda}{2}) & -i\sin(\frac{\lambda}{2}) & 0 \\
0 & -i\sin(\frac{\lambda}{2}) & \cos(\frac{\lambda}{2}) & 0 \\
i\sin(\frac{\lambda}{2}) & 0 & 0 & \cos(\frac{\lambda}{2}) \\
\end{bmatrix}
$$

[How to apply]
```python
Ryy(pi) | circuit([0, 1])
Ryy(pi) & [0, 1] | circuit
```

[Graph Representation]
```python
           ┌───────────┐
    q0:  ──┤0          ├──
           |  Ryy(pi)  |
    q1:  ──┤1          ├──
           └───────────┘
```

[QASM Representation]
```
ryy(pi) q[0], q[1]
```
"""

Rzz = BasicGate(*GATEINFO_MAP[GateType.rzz], is_original_gate=True)
r"""
Double-Qubits Z $\otimes$ Z Gate.

[Matrix Representation]

$$
Rzz_{q0,q1}(\lambda) = \begin{bmatrix}
e^{\frac{-i\lambda}{2}} & 0 & 0 & 0 \\ 
0 & e^{\frac{i\lambda}{2}} & 0 & 0 \\
0 & 0 & e^{\frac{i\lambda}{2}} & 0 \\
0 & 0 & 0 & e^{\frac{-i\lambda}{2}} \\
\end{bmatrix}
$$

[How to apply]
```python
Rzz(pi) | circuit([0, 1])
Rzz(pi) & [0, 1] | circuit
```

[Graph Representation]
```python
           ┌───────────┐
    q0:  ──┤0          ├──
           |  Rzz(pi)  |
    q1:  ──┤1          ├──
           └───────────┘
```

[QASM Representation]
```
rzz(pi) q[0], q[1]
```
"""

Rzx = BasicGate(*GATEINFO_MAP[GateType.rzx], is_original_gate=True)
r"""
Double-Qubits Z $\otimes$ X Gate.

[Matrix Representation]

$$
Rzx_{q0,q1}(\lambda) = \begin{bmatrix}
\cos(\frac{\lambda}{2}) & -i\sin(\frac{\lambda}{2}) & 0 & 0 \\ 
-i\sin(\frac{\lambda}{2}) & \cos(\frac{\lambda}{2}) & 0 & 0 \\
0 & 0 & \cos(\frac{\lambda}{2}) & i\sin(\frac{\lambda}{2}) \\
0 & 0 & i\sin(\frac{\lambda}{2}) & \cos(\frac{\lambda}{2}) \\
\end{bmatrix}
$$

[How to apply]
```python
Rzx(pi) | circuit([0, 1])
Rzx(pi) & [0, 1] | circuit
```

[Graph Representation]
```python
           ┌───────────┐
    q0:  ──┤0          ├──
           |  Rzx(pi)  |
    q1:  ──┤1          ├──
           └───────────┘
```

[QASM Representation]
```
rzx(pi) q[0], q[1]
```
"""

Measure = BasicGate(*GATEINFO_MAP[GateType.measure], is_original_gate=True)
"""
The Quantum Measurement Gate.

[How to apply]
```python
Measure | circuit(0)
Measure & 0 | circuit
```

[Graph Representation]
```python
       ┌───┐
     ──┤ M ├──
       └───┘
```

[QASM Representation]
```
measure q[0] -> c[0]
```
"""

Reset = BasicGate(*GATEINFO_MAP[GateType.reset], is_original_gate=True)
"""
The Quantum Reset Gate.

[How to apply]
```python
Reset | circuit(0)
Reset & 0 | circuit
```

[Graph Representation]
```python
      ┌───┐
    ──┤ R ├──
      └───┘
```

[QASM Representation]
```
reset q[0]
```
"""

Barrier = BasicGate(*GATEINFO_MAP[GateType.barrier], is_original_gate=True)
"""
The Barrier Gate.

[How to apply]
```python
Barrier | circuit(0)
Barrier & 0 | circuit
```

[Graph Representation]
```python
        ░
    ────░────
        ░
```

[QASM Representation]
```
barrier q[0]
```
"""

Swap = BasicGate(*GATEINFO_MAP[GateType.swap], is_original_gate=True)
r"""
Swap Gate.

[Matrix Representation]

$$
Swap_{q0,q1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 0 & 1 & 0 \\ 
0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 1 \\ 
\end{bmatrix}
$$

[How to apply]
```python
Swap | circuit([0, 1])
Swap & [0, 1] | circuit
```

[Graph Representation]
```python
           
    q0:  ─────X─────
              |
    q1:  ─────X─────
```

[QASM Representation]
```
swap q[0], q[1]
```
"""

iSwap = BasicGate(*GATEINFO_MAP[GateType.iswap], is_original_gate=True)
r"""
iSwap Gate, a double-qubit XX + YY Gate.

[Matrix Representation]

$$
iSwap_{q0,q1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 0 & i & 0 \\ 
0 & i & 0 & 0 \\ 
0 & 0 & 0 & 1 \\ 
\end{bmatrix}
$$

[How to apply]
```python
iSwap | circuit([0, 1])
iSwap & [0, 1] | circuit
```

[Graph Representation]
```python
           
    q0:  ─────X─────
              |
    q1:  ─────X─────
```

[QASM Representation]
```
iswap q[0], q[1]
```
"""

iSwap_dagger = BasicGate(*GATEINFO_MAP[GateType.iswapdg], is_original_gate=True)
r"""
iSwap Dagger Gate.

[Matrix Representation]

$$
iSwapdg_{q0,q1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & 0 & -i & 0 \\ 
0 & -i & 0 & 0 \\ 
0 & 0 & 0 & 1 \\ 
\end{bmatrix}
$$

[How to apply]
```python
iSwap_dagger | circuit([0, 1])
iSwap_dagger & [0, 1] | circuit
```

[Graph Representation]
```python
           
    q0:  ─────X─────
              |
    q1:  ─────X─────
```

[QASM Representation]
```
iswap_dagger q[0], q[1]
```
"""

sqiSwap = BasicGate(*GATEINFO_MAP[GateType.sqiswap], is_original_gate=True)
r"""
Swap Gate.

[Matrix Representation]

$$
sqiSwap_{q0,q1} = \begin{bmatrix}
1 & 0 & 0 & 0 \\ 
0 & \frac{1}{\sqrt{2}} & \frac{i}{\sqrt{2}} & 0 \\ 
0 & \frac{i}{\sqrt{2}} & \frac{1}{\sqrt{2}} & 0 \\ 
0 & 0 & 0 & 1 \\ 
\end{bmatrix}
$$

[How to apply]
```python
sqiSwap | circuit([0, 1])
sqiSwap & [0, 1] | circuit
```

[Graph Representation]
```python
           
    q0:  ─────X─────
              |
    q1:  ─────X─────
```

[QASM Representation]
```
sqiswap q[0], q[1]
```
"""

CCX = BasicGate(*GATEINFO_MAP[GateType.ccx], is_original_gate=True)
r"""
Double-Qubit Controlled X Gate, CCX Gate.

[Matrix Representation]

$$
CCX_{q0,q1,q2} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 
\end{bmatrix}
$$

[How to apply]
```python
CCX | circuit([0, 1, 2])
CCX & [0, 1, 2] | circuit
```

[Graph Representation]
```python
    q0:  ────■────
             |
             |
    q1:  ────■────
             |
           ┌───┐
    q2:  ──┤ X ├──
           └───┘
```

[QASM Representation]
```
ccx q[0], q[1], q[2]
```
"""

CCZ = BasicGate(*GATEINFO_MAP[GateType.ccz], is_original_gate=True)
r"""
Double-Qubit Controlled Z Gate, CCZ Gate.

[Matrix Representation]

$$
CCZ_{q0,q1,q2} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 0 & 0 & -1 \\ 
\end{bmatrix}
$$

[How to apply]
```python
CCZ | circuit([0, 1, 2])
CCZ & [0, 1, 2] | circuit
```

[Graph Representation]
```python
    q0:  ────■────
             |
             |
    q1:  ────■────
             |
           ┌───┐
    q2:  ──┤ Z ├──
           └───┘
```

[QASM Representation]
```
ccz q[0], q[1], q[2]
```
"""

CCRz = BasicGate(*GATEINFO_MAP[GateType.ccrz], is_original_gate=True)
r"""
Double-Qubit Controlled Rz Gate, CCRz Gate.

[Matrix Representation]

$$
CCRz_{q0,q1,q2}(\lambda) = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 0 & e^{\frac{-i\lambda}{2}} & 0 \\ 
0 & 0 & 0 & 0 & 0 & 0 & 0 & e^{\frac{i\lambda}{2}} \\ 
\end{bmatrix}
$$

[How to apply]
```python
CCRz(pi) | circuit([0, 1, 2])
CCRz(pi) & [0, 1, 2] | circuit
```

[Graph Representation]
```python
    q0:  ──────■───────
               |
               |
    q1:  ──────■───────
               |
           ┌────────┐
    q2:  ──┤ Rz(pi) ├──
           └────────┘
```

[QASM Representation]
```
ccrz(pi) q[0], q[1], q[2]
```
"""

CSwap = BasicGate(*GATEINFO_MAP[GateType.cswap], is_original_gate=True)
r"""
Controlled Swap Gate.

[Matrix Representation]

$$
CSwap_{q0,q1,q2} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 
\end{bmatrix}
$$

[How to apply]
```python
CSwap | circuit([0, 1, 2])
CSwap & [0, 1, 2] | circuit
```

[Graph Representation]
```python
    q0:  ──────■───────
               |
               |
    q1:  ──────X───────
               |
               |
    q2:  ──────X───────
```

[QASM Representation]
```
cswap q[0], q[1], q[2]
```
"""
