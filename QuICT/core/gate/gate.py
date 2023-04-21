from __future__ import annotations

from typing import Union
import numpy as np

from QuICT.core.utils import (
<<<<<<< HEAD
    GateType,
    MatrixType,
    SPECIAL_GATE_SET,
    DIAGONAL_GATE_SET,
    CGATE_LIST,
    PAULI_GATE_SET,
    CLIFFORD_GATE_SET,
    perm_decomposition,
    matrix_product_to_circuit,
=======
    matrix_product_to_circuit, GateType, MatrixType,
    CGATE_LIST, GATE_ARGS_MAP, PAULI_GATE_SET, CLIFFORD_GATE_SET, GATEINFO_MAP
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
)
from QuICT.tools.exception.core import (
    TypeError,
    ValueError,
    GateAppendError,
    GateQubitAssignedError,
    QASMError,
    GateMatrixError,
    GateParametersAssignedError,
)

from .utils import Variable, GateMatrixGenerator, ComplexGateBuilder, InverseGate


class BasicGate(object):
    """the abstract SuperClass of all basic quantum gate

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
<<<<<<< HEAD

        qasm_name(str, read only): gate's name in the OpenQASM 2.0
        type(GateType, read only): gate's type described by GateType

        matrix(np.array): the unitary matrix of the quantum gate act on targets
        required_grad(bool): True for required grad,false for non required grad
    """

=======
    """
    ####################################################################
    ############          Quantum Gate's property           ############
    ####################################################################
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
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

<<<<<<< HEAD
    @controls.setter
    def controls(self, controls: int):
        assert isinstance(controls, int), TypeError(
            "BasicGate.controls", "int", type(controls)
        )
        self._controls = controls
=======
    def get_matrix(self, precision) -> np.ndarray:
        return GateMatrixGenerator().get_matrix(self, precision)
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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

<<<<<<< HEAD
        assert len(cargs) == len(set(cargs)), ValueError(
            "BasicGate.cargs", "not have duplicated value", cargs
        )
        self._cargs = cargs
=======
        return self._grad_matrix
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

    ################    Quantum Gate's Target Qubits    ################
    @property
    def targets(self) -> int:
        return self._targets

<<<<<<< HEAD
    @targets.setter
    def targets(self, targets: int):
        assert isinstance(targets, int), TypeError(
            "BasicGate.targets", "int", type(targets)
        )
        self._targets = targets
=======
    @property
    def targ(self):
        return self.targs[0]
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

    @property
    def targs(self):
        return self._targs

    @targs.setter
    def targs(self, targs: list):
        if isinstance(targs, int):
            targs = [targs]

        assert len(targs) == len(set(targs)), ValueError(
            "BasicGate.targs", "not have duplicated value", targs
        )
        assert not set(self._cargs) & set(targs), ValueError(
            "BasicGate.targs",
            "have no same index with control qubits",
            set(self._cargs) & set(targs),
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

<<<<<<< HEAD
        if len(self._pargs) != self.params:
            raise ValueError(
                "BasicGate.pargs:length",
                f"equal to gate's parameter number {self._pargs}",
                len(pargs),
            )
=======
        assert len(cargs) == len(set(cargs)), ValueError("BasicGate.cargs", "not have duplicated value", cargs)
        self._cargs = cargs

    #################    Quantum Gate's parameters    ##################
    @property
    def params(self) -> int:
        return self._params
    
    @property
    def variables(self) -> int:
        return self._variables
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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

<<<<<<< HEAD
    @property
    def qasm_name(self):
        return self._qasm_name
    
    
=======
        self.permit_element(pargs)
        assert len(self._pargs) == self.params, \
            ValueError("BasicGate.pargs:length", f"equal to gate's parameter number {self._pargs}", len(pargs))

        self._pargs = pargs
        self._is_matrix_update = True
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

    def __init__(
        self,
        controls: int,
        targets: int,
        params: int,
        type_: GateType,
        matrix_type: MatrixType = MatrixType.normal,
<<<<<<< HEAD
        requires_grad:bool=False
=======
        pargs: list = [],
        precision: str = "double",
        is_original_gate: bool = False
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
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
<<<<<<< HEAD
        self._targets = targets
        self._params = params
        self._cargs = []  # list of int
        self._targs = []  # list of int
        self._pargs = []  # list of float/..
        self._requires_grad = requires_grad

        assert isinstance(type_, GateType), TypeError(
            "BasicGate.type", "GateType", type(type_)
        )
=======
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
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
        self._type = type_
        self._matrix_type = matrix_type

<<<<<<< HEAD
        self.assigned_qubits = []  # list of qubits
=======
        assert precision in ["double", "single"], \
            ValueError("BasicGate.precision", "not within [double, single]", precision)
        self._precision = precision
        self._matrix = None
        self._target_matrix = None
        self._grad_matrix = None
        self._qasm_name = str(type_.name)
        self.assigned_qubits = []   # list of qubits' id
        self._is_matrix_update = False
        self._is_original = is_original_gate
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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
<<<<<<< HEAD
            raise GateAppendError(
                f"Failure to append gate {self.name} to targets, due to {e}"
            )
=======
            raise GateAppendError(f"Failure to append gate {self} to targets, due to {e}")
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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
<<<<<<< HEAD
            assert _gate.is_single(), GateQubitAssignedError(
                "The qubits number should equal to the quantum gate."
            )

            _gate.targs = [targets]
        elif isinstance(targets, list):
            if len(targets) != _gate.controls + _gate.targets:
                raise GateQubitAssignedError(
                    "The qubits number should equal to the quantum gate."
                )

            _gate.cargs = targets[: _gate.controls]
            _gate.targs = targets[_gate.controls :]
=======
            targets = [targets]

        if not isinstance(targets, list):
            raise TypeError("BasicGate.&", "int or list<int>", type(targets))

        assert len(targets) == self.controls + self.targets, \
            GateQubitAssignedError("The qubits number should equal to the quantum gate.")

        if self._is_original:
            _gate = self.copy()
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
        else:
            _gate = self

        _gate.cargs = targets[:_gate.controls]
        _gate.targs = targets[_gate.controls:]

        if CGATE_LIST:
            CGATE_LIST[-1].append(_gate)

<<<<<<< HEAD
    def __call__(self,requires_grad:bool=False):
        """give parameters for the gate, and give parameters by "()", and parameters should be one of int/float/complex
=======
        return _gate

    def __call__(self, *args):
        """ give parameters for the gate, and give parameters by "()", and parameters should be one of int/float/complex
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        Some Examples are like this:

        Rz(np.pi / 2)
        U3(np.pi / 2, 0, 0)

        *Important*: There is no parameters for some quantum gate.

        Returns:
            BasicGate: the gate after filled by parameters
        """
<<<<<<< HEAD
        self._requires_grad=requires_grad
        return self.copy()
    
=======
        assert len(args) == self.params, \
            GateParametersAssignedError("The number of given parameters not matched the quantum gate.")

        if self._is_original:
            _gate = self.copy()
        else:
            _gate = self

        _gate.pargs = list(args)
        return _gate
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

    def __eq__(self, other):
        assert isinstance(other, BasicGate), TypeError(
            "BasicGate.==", "BasicGate", type(other)
        )
        if (
<<<<<<< HEAD
            self.type != other.type
            or (self.cargs + self.targs) != (other.cargs + other.targs)
            or not np.allclose(self.matrix, other.matrix)
=======
            self.type != other.type or
            self.matrix_type != other.matrix_type or
            (self.cargs + self.targs) != (other.cargs + other.targs) or
            self.pargs != other.pargs
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
        ):
            return False

        if self.type == GateType.unitary and not np.allclose(self.matrix, other.matrix):
            return False

        return True

<<<<<<< HEAD
    def update_name(self, qubit_id: str, circuit_idx: int = None):
        """Updated gate's name with the given information

        Args:
            qubit_id (str): The qubit's unique ID.
            circuit_idx (int, optional): The gate's order index in the circuit. Defaults to None.
        """
        qubit_id = qubit_id[:6]
        name_parts = self.name.split("-")
        name_parts[1] = qubit_id
=======
    def __str__(self):
        gstr = f"gate type: {self.type}; "
        if self.params > 0:
            gstr += f"parameters number: {self.params}; parameters: {self.pargs}; "

        if self.controls > 0:
            gstr += f"controls qubits' number: {self.controls}; controls qubits' indexes: {self.cargs}; "
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        gstr += f"target qubits' number: {self.targets}; target qubits' indexes: {self.targs}"

<<<<<<< HEAD
        self.name = "-".join(name_parts)

    def __str__(self):
        """get gate information"""
        gate_info = {
            "name": self.name,
            "controls": self.controls,
            "control_bit": self.cargs,
            "targets": self.targets,
            "target_bit": self.targs,
            "parameters": self.pargs,
        }

        return str(gate_info)

    def qasm(self):
        """generator OpenQASM string for the gate
=======
        return gstr

    def qasm(self, targs: list = None):
        """ generator OpenQASM string for the gate
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        Return:
            string: the OpenQASM 2.0 describe of the gate
        """
        if self.type in [GateType.perm, GateType.perm_fx, GateType.qft, GateType.iqft]:
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

<<<<<<< HEAD
        ctargs = [f"q[{ctarg}]" for ctarg in self.cargs + self.targs]
        ctargs_string = " " + ", ".join(ctargs) + ";\n"
=======
        qubit_idxes = self.cargs + self.targs if targs is None else targs
        ctargs = [f"q[{ctarg}]" for ctarg in qubit_idxes]
        ctargs_string = " " + ', '.join(ctargs) + ";\n"
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
        qasm_string += ctargs_string

        return qasm_string

<<<<<<< HEAD
    def convert_precision(self):
        """Convert gate's precision into single precision np.complex64."""
        if self.type in [GateType.measure, GateType.reset, GateType.barrier]:
            return

        self._precision = (
            np.complex64 if self._precision == np.complex128 else np.complex128
        )
        if self.params == 0:
            self._matrix = self.matrix.astype(self._precision)

    def inverse(self):
        """the inverse of the gate
=======
    def inverse(self):
        """ the inverse of the quantum gate, if there is no inverse gate, return itself.
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        Return:
            BasicGate: the inverse of the gate
        """
        inverse_gargs, inverse_pargs = InverseGate.get_inverse_gate(self.type, self.pargs)

        # Deal with inverse_gargs
        if inverse_gargs is None:
            return self

        return gate_builder(inverse_gargs, params=inverse_pargs)

    def build_gate(self):
        """ Gate Decomposition, which divided the current gate with a set of small gates. """
        if self.type == GateType.cu3:
            return ComplexGateBuilder.build_gate(self.type, self.parg, self.matrix)

        gate_list = ComplexGateBuilder.build_gate(self.type, self.parg)
        if gate_list is None:
            return None

        cgate = self._cgate_generator_from_build_gate(gate_list)
        gate_args = self.cargs + self.targs
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

<<<<<<< HEAD
    def commutative(self, goal, eps=1e-7):
        """decide whether gate is commutative with another gate
=======
    def commutative(self, goal: BasicGate, eps=1e-7):
        """ decide whether gate is commutative with another gate
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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
        if (self.is_special() and self.type != GateType.unitary) or (
            goal.is_special() and self.type != GateType.unitary
        ):
            return False

        # It means commuting that any of the target matrices is close to identity
<<<<<<< HEAD
        if np.allclose(
            A, np.identity(1 << self.targets), rtol=eps, atol=eps
        ) or np.allclose(B, np.identity(1 << goal.targets), rtol=eps, atol=eps):
=======
        if (self.is_identity() or goal.is_identity()):
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
            return True

        # Check the target matrices of the gates
        A = self.target_matrix
        B = goal.target_matrix
        # For gates whose number of target qubits is 1, optimized judgment could be used
        if self.targets == 1 and goal.targets == 1:
            # Diagonal target gates commutes with the control qubits
            if (len(self_controls & goal_targets) > 0 and not goal.is_diagonal()) or (
                len(goal_controls & self_targets) > 0 and not self.is_diagonal()
            ):
                return False
            # Compute the target matrix commutation
            if len(goal_targets & self_targets) > 0 and not np.allclose(
                A.dot(B), B.dot(A), rtol=eps, atol=eps
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
            return np.allclose(
                self_matrix.dot(goal_matrix),
                goal_matrix.dot(self_matrix),
                rtol=eps,
                atol=eps,
            )

    def is_single(self) -> bool:
        """judge whether gate is a one qubit gate(excluding special gate like measure, reset, custom and so on)

        Returns:
            bool: True if it is a one qubit gate
        """
        return self.targets + self.controls == 1

    def is_control_single(self) -> bool:
        """judge whether gate has one control bit and one target bit

        Returns:
            bool: True if it is has one control bit and one target bit
        """
        return self.controls == 1 and self.targets == 1

    def is_clifford(self) -> bool:
        """judge whether gate's matrix is a Clifford gate

        Returns:
            bool: True if gate's matrix is a Clifford gate
        """
        return self.type in CLIFFORD_GATE_SET

    def is_diagonal(self) -> bool:
        """judge whether gate's matrix is diagonal

        Returns:
            bool: True if gate's matrix is diagonal
        """
<<<<<<< HEAD
        return self.type in DIAGONAL_GATE_SET or (
            self.type == GateType.unitary and self._is_diagonal()
        )

    def _is_diagonal(self) -> bool:
        return np.allclose(np.diag(np.diag(self.matrix)), self.matrix)
=======
        return self.matrix_type in [MatrixType.diagonal, MatrixType.control]
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

    def is_pauli(self) -> bool:
        """judge whether gate's matrix is a Pauli gate

        Returns:
            bool: True if gate's matrix is a Pauli gate
        """
        return self.type in PAULI_GATE_SET

    def is_special(self) -> bool:
<<<<<<< HEAD
        """judge whether gate's is special gate, which is one of
        [Measure, Reset, Barrier, Perm, Unitary, ...]
=======
        """ judge whether gate's is special gate, which is one of
        [Measure, Reset, Barrier, Perm, ...]
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        Returns:
            bool: True if gate's matrix is special
        """
        return self.matrix_type == MatrixType.special

    def is_identity(self) -> bool:
        """ judge whether gate's matrix is identity matrix

<<<<<<< HEAD
        return np.allclose(
            self.matrix,
            np.identity(1 << (self.controls + self.targets), dtype=self.precision),
        )

    def expand(self, qubits: Union[int, list]) -> bool:
        """expand self matrix into the circuit's unitary linear space. If input qubits is integer, please make sure
=======
        Returns:
            bool: True if gate's matrix is identity
        """
        return self.type == GateType.id or self.matrix_type == MatrixType.identity

    def expand(self, qubits: Union[int, list], device: str = "CPU") -> bool:
        """ expand self matrix into the circuit's unitary linear space. If input qubits is integer, please make sure
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
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
        if len(gate_args) == 0:  # Deal with not assigned quantum gate
            gate_args = [qubits[i] for i in range(self.controls + self.targets)]

        updated_args = [qubits.index(garg) for garg in gate_args]
        return matrix_product_to_circuit(self.matrix, updated_args, qubits_num, device)

    def copy(self):
        """return a copy of this gate

        Returns:
            gate(BasicGate): a copy of this gate
        """
        if self.variables == 0:
            pargs = self.pargs
        else:
            pargs = [parg.copy() for parg in self.pargs]
        gate = BasicGate(
            self.controls, self.targets, self.params, self.type,
            self.matrix_type, pargs, self.precision
        )

        if len(self.targs) > 0:
            gate.targs = self.targs[:]

<<<<<<< HEAD
        gate.pargs = copy.deepcopy(self.pargs)
        gate.targs = copy.deepcopy(self.targs)
        gate.cargs = copy.deepcopy(self.cargs)
        gate._requires_grad = copy.deepcopy(self._requires_grad)
=======
        if self.controls > 0 and len(self.cargs) > 0:
            gate.cargs = self.cargs[:]
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        if self.assigned_qubits:
            gate.assigned_qubits = self.assigned_qubits[:]

        return gate

    def permit_element(self, element):
        """judge whether the type of a parameter is int/float/complex

        for a quantum gate, the parameter should be int/float/complex

        Args:
            element: the element to be judged

        Returns:
            bool: True if the type of element is int/float/complex
        """
        if not isinstance(element, list):
            element = [element]

<<<<<<< HEAD
            raise TypeError(self.type, "int/float/complex", type(element))
    
   
    def is_requires_grad(self):
        return self._requires_grad
    def set_requires_grad(self,requires_grad:bool):
        self._requires_grad = requires_grad
        return 


class HGate(BasicGate):
    """Hadamard gate"""

    def __init__(self):
        super().__init__(controls=0, targets=1, params=0, type_=GateType.h)

        self.matrix = np.array(
            [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
            dtype=self._precision,
        )
=======
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
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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
        if isinstance(matrix, list):
            matrix = np.array(matrix)

<<<<<<< HEAD
class HYGate(BasicGate):
    """Self-inverse gate"""

    def __init__(self):
        super().__init__(controls=0, targets=1, params=0, type_=GateType.hy)

        self.matrix = np.array(
            [[1 / np.sqrt(2), -1j / np.sqrt(2)], [1j / np.sqrt(2), -1 / np.sqrt(2)]],
            dtype=self._precision,
        )
=======
        # Validate precision
        assert np.issubdtype(matrix.dtype, np.complex128) or np.issubdtype(matrix.dtype, np.complex64), \
            TypeError("unitary.matrix.dtype", "complex64/128", matrix.dtype)
        precision = "double" if matrix.dtype == np.complex64 else "single"

        # Validate matrix shape is square
        length, width = matrix.shape
        if length != width:
            N = int(np.log2(matrix.size))
            assert N ^ 2 == matrix.size, GateMatrixError("the shape of unitary matrix should be square.")
            matrix = matrix.reshape(N, N)
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        n = int(np.log2(matrix.shape[0]))
        if (1 << n) != matrix.shape[0]:
            raise GateMatrixError("the length of list should be the square of power(2, n)")

        if matrix_type is None:
            matrix_type, controls = self.validate_matrix_type(matrix)
        else:
            matrix_type = matrix_type
            controls = 0

        if controls == n:
            controls = 0

<<<<<<< HEAD
class SGate(BasicGate):
    """S gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.s,
            matrix_type=MatrixType.control,
=======
        super().__init__(
            controls=controls, targets=(n - controls), params=0,
            type_=GateType.unitary, matrix_type=matrix_type, precision=precision
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
        )
        self._matrix = matrix

<<<<<<< HEAD
        self.matrix = np.array([[1, 0], [0, 1j]], dtype=self._precision)

    def inverse(self):
        """change it be sdg gate"""
        _Sdagger = SDaggerGate()
        _Sdagger.targs = copy.deepcopy(self.targs)
        _Sdagger.assigned_qubits = copy.deepcopy(self.assigned_qubits)
=======
    @staticmethod
    def validate_matrix_type(matrix: np.ndarray) -> MatrixType:
        """ Check the matrix's type about given unitary matrix

        Args:
            matrix (np.ndarry): The given unitary matrix
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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

<<<<<<< HEAD
class SDaggerGate(BasicGate):
    """The conjugate transpose of Phase gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.sdg,
            matrix_type=MatrixType.control,
        )

        self.matrix = np.array([[1, 0], [0, -1j]], dtype=self._precision)

    def inverse(self):
        """change it to be s gate"""
        _Sgate = SGate()
        _Sgate.targs = copy.deepcopy(self.targs)
        _Sgate.assigned_qubits = copy.deepcopy(self.assigned_qubits)

        return _Sgate


S_dagger = SDaggerGate()


class XGate(BasicGate):
    """Pauli-X gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.x,
            matrix_type=MatrixType.swap,
        )

        self.matrix = np.array([[0, 1], [1, 0]], dtype=self._precision)


X = XGate()


class YGate(BasicGate):
    """Pauli-Y gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.y,
            matrix_type=MatrixType.reverse,
        )

        self.matrix = np.array([[0, -1j], [1j, 0]], dtype=self._precision)
=======
        return matrix_type, controls

    def build_gate(self):
        return ComplexGateBuilder.build_unitary(self._matrix)

    def inverse(self):
        inverse_matrix = np.asmatrix(self.matrix).H

        return Unitary(inverse_matrix)

    def copy(self):
        _gate = Unitary(self.matrix, self.matrix_type)
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        if len(self.targs) > 0:
            _gate.targs = self.targs[:]

        if self.assigned_qubits:
            _gate.assigned_qubits = self.assigned_qubits[:]

        return _gate

<<<<<<< HEAD
class ZGate(BasicGate):
    """Pauli-Z gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.z,
            matrix_type=MatrixType.control,
        )

        self.matrix = np.array([[1, 0], [0, -1]], dtype=self._precision)
=======

class Perm(BasicGate):
    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = self._build_matrix()
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

        return self._matrix

    @property
    def target_matrix(self):
        return self.matrix

    def __init__(self, targets: int, params: list):
        """
        Args:
            n (int): the number of target qubits
            params (list[int]): the list of index, and the index represent which should be 1.

<<<<<<< HEAD
class SXGate(BasicGate):
    """sqrt(X) gate"""

    def __init__(self):
        super().__init__(controls=0, targets=1, params=0, type_=GateType.sx)

        self.matrix = np.array(
            [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=self._precision
        )

    def inverse(self):
        """change it be rx gate"""
        _Rx = RxGate([-np.pi / 2])
        _Rx.targs = copy.deepcopy(self.targs)
=======
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
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd

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

<<<<<<< HEAD
class SYGate(BasicGate):
    """sqrt(Y) gate"""

    def __init__(self):
        super().__init__(controls=0, targets=1, params=0, type_=GateType.sy)

        self.matrix = np.array(
            [[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]],
            dtype=self._precision,
        )

    def inverse(self):
        """change it to be ry gate"""
        _Ry = RyGate([-np.pi / 2])
        _Ry.targs = copy.deepcopy(self.targs)

        return _Ry


SY = SYGate()


class SWGate(BasicGate):
    """sqrt(W) gate"""

    def __init__(self):
        super().__init__(controls=0, targets=1, params=0, type_=GateType.sw)

        self.matrix = np.array(
            [[1 / np.sqrt(2), -np.sqrt(1j / 2)], [np.sqrt(-1j / 2), 1 / np.sqrt(2)]],
            dtype=self._precision,
        )

    def inverse(self):
        """change it be U2 gate"""
        _U2 = U2Gate([3 * np.pi / 4, 5 * np.pi / 4])
        _U2.targs = copy.deepcopy(self.targs)
=======

        return matrix_

    def inverse(self):
        inverse_targs = [self.targets - 1 - t for t in self.pargs]

        return Perm(self.targets, inverse_targs)
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd


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

<<<<<<< HEAD
class IDGate(BasicGate):
    """Identity gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.id,
            matrix_type=MatrixType.diagonal,
        )

        self.matrix = np.array([[1, 0], [0, 1]], dtype=self._precision)
=======
        targets = targets + 1
        parameters = 1 << targets
        pargs = []
        for idx in range(1 << targets):
            if idx >> 1 in params:
                pargs.append(idx ^ 1)
            else:
                pargs.append(idx)

        super().__init__(0, targets, parameters, GateType.perm_fx, MatrixType.normal, pargs)
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd


def gate_builder(gate_type, precision: str = "double", params: list = [], random_params: bool = False) -> BasicGate:
    """ Build the target Quantum Gate.

    Args:
        gate_type (_type_): The gate's type
        precision (str, optional): The gate's precision. Defaults to "double".
        params (list, optional): The gate's parameters. Defaults to [].
        random_params (bool, optional): Whether using random parameters. Defaults to False.

<<<<<<< HEAD
class U1Gate(BasicGate):
    """Diagonal single-qubit gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type_=GateType.u1,
            matrix_type=MatrixType.control,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return U1Gate([alpha])

    @property
    def matrix(self):
        return np.array(
            [[1, 0], [0, np.exp(1j * self.pargs[0])]], dtype=self._precision
        )

    def inverse(self):
        _U1 = self.copy()
        _U1.pargs = [-self.pargs[0]]

        return _U1


U1 = U1Gate()


class U2Gate(BasicGate):
    """One-pulse single-qubit gate"""

    def __init__(self, params: list = [np.pi / 2, np.pi / 2]):
        super().__init__(controls=0, targets=1, params=2, type_=GateType.u2)

        self.pargs = params

    def __call__(self, alpha, beta):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate
            beta (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        params = [alpha, beta]

        for param in params:
            self.permit_element(param)

        return U2Gate(params)

    @property
    def matrix(self):
        sqrt2 = 1 / np.sqrt(2)
        return np.array(
            [
                [1 * sqrt2, -np.exp(1j * self.pargs[1]) * sqrt2],
                [
                    np.exp(1j * self.pargs[0]) * sqrt2,
                    np.exp(1j * (self.pargs[0] + self.pargs[1])) * sqrt2,
                ],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _U2 = self.copy()
        _U2.pargs = [np.pi - self.pargs[1], np.pi - self.pargs[0]]

        return _U2


U2 = U2Gate()


class U3Gate(BasicGate):
    """Two-pulse single-qubit gate"""

    def __init__(self, params: list = [0, 0, np.pi / 2]):
        super().__init__(controls=0, targets=1, params=3, type_=GateType.u3)

        self.pargs = params

    def __call__(self, alpha, beta, gamma):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate
            beta (int/float/complex): The parameter for gate
            gamma (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        params = [alpha, beta, gamma]

        for param in params:
            self.permit_element(param)

        return U3Gate(params)

    @property
    def matrix(self):
        return np.array(
            [
                [
                    np.cos(self.pargs[0] / 2),
                    -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2),
                ],
                [
                    np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
                    np.exp(1j * (self.pargs[1] + self.pargs[2]))
                    * np.cos(self.pargs[0] / 2),
                ],
            ],
            dtype=self._precision,
        )
    @property
    def  parti_deri_adj(self):
        return np.array(
            [
            [
                [
                    -np.sin(self.pargs[0] / 2)/2,
                    np.exp(-1j * self.pargs[1]) * np.cos(self.pargs[0] / 2)/2,
                ],
                [
                    -np.exp(-1j * self.pargs[2]) * np.cos(self.pargs[0] / 2)/2,
                    -np.exp(-1j * (self.pargs[1] + self.pargs[2]))
                    * np.cos(self.pargs[0] / 2)/2,
                ],
            ],
            [
                [
                    0,
                    -np.exp(-1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
                ],
                [
                    0,
                    -np.exp(-1j * (self.pargs[1] + self.pargs[2]))
                    * np.cos(self.pargs[0] / 2),
                ],
            ],
            [
                [
                    0,
                    0,
                ],
                [
                    np.exp(-1j * self.pargs[2]) * np.sin(self.pargs[0] / 2),
                    -np.exp(-1j * (self.pargs[1] + self.pargs[2]))
                    * np.cos(self.pargs[0] / 2),
                ],
            ],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _U3 = self.copy()
        _U3.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]

        return _U3


U3 = U3Gate()


class RxGate(BasicGate):
    """Rotation around the x-axis gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(controls=0, targets=1, params=1, type_=GateType.rx)

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return RxGate([alpha])

    @property
    def matrix(self):
        return np.array(
            [
                [np.cos(self.parg / 2), 1j * -np.sin(self.parg / 2)],
                [1j * -np.sin(self.parg / 2), np.cos(self.parg / 2)],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _Rx = self.copy()
        _Rx.pargs = [-self.pargs[0]]

        return _Rx


Rx = RxGate()


class RyGate(BasicGate):
    """Rotation around the y-axis gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(controls=0, targets=1, params=1, type_=GateType.ry)

        self.pargs = params

    def __call__(self, alpha,):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate
            requires_grad(bool): tag of gate indicates wheather update is needed

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return RyGate([alpha])

    @property
    def matrix(self):
        return np.array(
            [
                [np.cos(self.pargs[0] / 2), -np.sin(self.pargs[0] / 2)],
                [np.sin(self.pargs[0] / 2), np.cos(self.pargs[0] / 2)],
            ],
            dtype=self._precision,
        )
    @property
    def  parti_deri_adj(self):
        return np.array(
            [
                [-np.sin(self.pargs[0] / 2)/2, np.cos(self.pargs[0] / 2)/2],
                [-np.cos(self.pargs[0] / 2)/2, -np.sin(self.pargs[0] / 2)/2],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _Ry = self.copy()
        _Ry.pargs = [-self.pargs[0]]

        return _Ry


Ry = RyGate()


class RzGate(BasicGate):
    """Rotation around the z-axis gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type_=GateType.rz,
            matrix_type=MatrixType.diagonal,
        )

        self.pargs = params

    def __call__(self, alpha,requires_grad:bool=False):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate
            requires_grad(bool): tag of gate indicates wheather update is needed

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)
        self._requires_grad=requires_grad

        return RzGate([alpha])

    @property
    def matrix(self):
        return np.array(
            [[np.exp(-self.parg / 2 * 1j), 0], [0, np.exp(self.parg / 2 * 1j)]],
            dtype=self._precision,
        )
    @property
    def  parti_deri_adj(self):
        return np.array(
            [
            [[np.exp(self.parg / 2 * 1j)/2, 0], [0, -np.exp(-self.parg / 2 * 1j)/2]],
            
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _Rz = self.copy()
        _Rz.pargs = [-self.pargs[0]]

        return _Rz


Rz = RzGate()


class TGate(BasicGate):
    """T gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.t,
            matrix_type=MatrixType.control,
        )

        self.matrix = np.array(
            [[1, 0], [0, 1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)]], dtype=self._precision
        )

    def inverse(self):
        """change it be tdg gate"""
        _Tdagger = TDaggerGate()
        _Tdagger.targs = copy.deepcopy(self.targs)
        _Tdagger.assigned_qubits = copy.deepcopy(self.assigned_qubits)

        return _Tdagger


T = TGate()


class TDaggerGate(BasicGate):
    """The conjugate transpose of T gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.tdg,
            matrix_type=MatrixType.control,
        )

        self.matrix = np.array(
            [[1, 0], [0, 1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)]], dtype=self._precision
        )

    def inverse(self):
        """change it to be t gate"""
        _Tgate = TGate()
        _Tgate.targs = copy.deepcopy(self.targs)
        _Tgate.assigned_qubits = copy.deepcopy(self.assigned_qubits)

        return _Tgate


T_dagger = TDaggerGate()


class PhaseGate(BasicGate):
    """Phase gate"""

    def __init__(self, params: list = [0]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type_=GateType.phase,
            matrix_type=MatrixType.control,
        )

        self.pargs = params
        self._qasm_name = "p"

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return PhaseGate([alpha])

    @property
    def matrix(self):
        return np.array([[1, 0], [0, np.exp(self.parg * 1j)]], dtype=self._precision)

    def inverse(self):
        _Phase = self.copy()
        _Phase.pargs = [-self.parg]

        return _Phase


Phase = PhaseGate()


class GlobalPhaseGate(BasicGate):
    """Phase gate"""

    def __init__(self, params: list = [0]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type_=GateType.gphase,
            matrix_type=MatrixType.diagonal,
        )
        self._qasm_name = "phase"
        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return GlobalPhaseGate([alpha])

    @property
    def matrix(self):
        return np.array(
            [[np.exp(self.parg * 1j), 0], [0, np.exp(self.parg * 1j)]],
            dtype=self._precision,
        )

    def inverse(self):
        _Phase = self.copy()
        _Phase.pargs = [-self.parg]

        return _Phase


GPhase = GlobalPhaseGate()


class CZGate(BasicGate):
    """controlled-Z gate"""

    def __init__(self):
        super().__init__(
            controls=1,
            targets=1,
            params=0,
            type_=GateType.cz,
            matrix_type=MatrixType.control,
        )

        self.matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
            dtype=self._precision,
        )

        self._target_matrix = np.array([[1, 0], [0, -1]], dtype=self._precision)

    @property
    def target_matrix(self):
        return self._target_matrix


CZ = CZGate()


class CXGate(BasicGate):
    """controlled-X gate"""

    def __init__(self):
        super().__init__(
            controls=1,
            targets=1,
            params=0,
            type_=GateType.cx,
            matrix_type=MatrixType.reverse,
        )

        self.matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=self._precision,
        )

        self._target_matrix = np.array([[0, 1], [1, 0]], dtype=self._precision)

    @property
    def target_matrix(self):
        return self._target_matrix


CX = CXGate()


class CYGate(BasicGate):
    """controlled-Y gate"""

    def __init__(self):
        super().__init__(
            controls=1,
            targets=1,
            params=0,
            type_=GateType.cy,
            matrix_type=MatrixType.reverse,
        )

        self.matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
            dtype=self._precision,
        )

        self._target_matrix = np.array([[0, -1j], [1j, 0]], dtype=self._precision)

    @property
    def target_matrix(self):
        return self._target_matrix


CY = CYGate()


class CHGate(BasicGate):
    """controlled-Hadamard gate"""

    def __init__(self):
        super().__init__(controls=1, targets=1, params=0, type_=GateType.ch)

        self.matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
            ],
            dtype=self._precision,
        )

        self._target = np.array(
            [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
            dtype=self._precision,
        )

    @property
    def target_matrix(self):
        return self._target_matrix


CH = CHGate()


class CRxGate(BasicGate):
    """controlled-Rx gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=1,
            targets=1,
            params=1,
            type_=GateType.crx,
            matrix_type=MatrixType.diag_normal,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return CRxGate([alpha])

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(self.parg / 2), -np.sin(self.parg / 2) * 1j],
                [0, 0, -np.sin(self.parg / 2) * 1j, np.cos(self.parg / 2)],
            ],
            dtype=self._precision,
        )


CRx = CRxGate()


class CRyGate(BasicGate):
    """controlled-Ry gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=1,
            targets=1,
            params=1,
            type_=GateType.cry,
            matrix_type=MatrixType.diag_normal,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return CRyGate([alpha])

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.cos(self.parg / 2), -np.sin(self.parg / 2)],
                [0, 0, np.sin(self.parg / 2), np.cos(self.parg / 2)],
            ],
            dtype=self._precision,
        )


CRy = CRyGate()


class CRzGate(BasicGate):
    """controlled-Rz gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=1,
            targets=1,
            params=1,
            type_=GateType.crz,
            matrix_type=MatrixType.diagonal,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return CRzGate([alpha])

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, np.exp(-self.parg / 2 * 1j), 0],
                [0, 0, 0, np.exp(self.parg / 2 * 1j)],
            ],
            dtype=self._precision,
        )

    @property
    def target_matrix(self) -> np.ndarray:
        return np.array(
            [[np.exp(-self.parg / 2 * 1j), 0], [0, np.exp(self.parg / 2 * 1j)]],
            dtype=self._precision,
        )

    def inverse(self):
        _CRz = self.copy()
        _CRz.pargs = [-self.pargs[0]]

        return _CRz


CRz = CRzGate()


class CU1Gate(BasicGate):
    """Controlled-U1 gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=1,
            targets=1,
            params=1,
            type_=GateType.cu1,
            matrix_type=MatrixType.control,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return CU1Gate([alpha])

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(1j * self.pargs[0])],
            ],
            dtype=self._precision,
        )

    @property
    def target_matrix(self) -> np.ndarray:
        return np.array(
            [[1, 0], [0, np.exp(1j * self.pargs[0])]], dtype=self._precision
        )

    def inverse(self):
        _CU1 = self.copy()
        _CU1.pargs = [-self.pargs[0]]

        return _CU1

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            CRz(self.parg) & [0, 1]
            U1(self.parg / 2) & 0

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


CU1 = CU1Gate()


class CU3Gate(BasicGate):
    """Controlled-U3 gate"""

    def __init__(self, params: list = [np.pi / 2, 0, 0]):
        super().__init__(controls=1, targets=1, params=3, type_=GateType.cu3)

        self.pargs = params

    def __call__(self, alpha, beta, gamma):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate
            beta (int/float/complex): The parameter for gate
            gamma (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        params = [alpha, beta, gamma]

        for param in params:
            self.permit_element(param)

        return CU3Gate(params)

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [
                    0,
                    0,
                    np.cos(self.pargs[0] / 2),
                    -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2),
                ],
                [
                    0,
                    0,
                    np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
                    np.exp(1j * (self.pargs[1] + self.pargs[2]))
                    * np.cos(self.pargs[0] / 2),
                ],
            ],
            dtype=self._precision,
        )
    @property
    def parti_deri_adj(self):
        return np.array(
            [
            [
                [-np.sin(self.pargs[0] / 2)/2,
                 np.exp(-1j * self.pargs[1]) * np.cos(self.pargs[0] / 2)/2],
                [-np.exp(-1j * self.pargs[2]) * np.cos(self.pargs[0] / 2)/2,
                 -np.exp(-1j * (self.pargs[1] + self.pargs[2]))
                    * np.sin(self.pargs[0] / 2)/2,
                ],
            ],
             [
                [0, 
                 -np.exp(-1j * self.pargs[1]) * np.sin(self.pargs[0] / 2)],
                [0,
                 -np.exp(-1j * (self.pargs[1] + self.pargs[2]))* np.cos(self.pargs[0] / 2),
                ],
            ],
             [
                [ 0, 0],
                [
                 np.exp(-1j * self.pargs[2]) * np.sin(self.pargs[0] / 2),
                 -np.exp(-1j * (self.pargs[1] + self.pargs[2]))* np.cos(self.pargs[0] / 2),
                ],
            ],
            ],
            dtype=self._precision,
        )

    @property
    def target_matrix(self) -> np.ndarray:
        return np.array(
            [
                [
                    np.cos(self.pargs[0] / 2),
                    -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2),
                ],
                [
                    np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
                    np.exp(1j * (self.pargs[1] + self.pargs[2]))
                    * np.cos(self.pargs[0] / 2),
                ],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _CU3 = self.copy()
        _CU3.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]

        return _CU3

    def build_gate(self):
        from QuICT.qcda.synthesis import UnitaryDecomposition

        assert self.controls + self.targets > 0
        mapping_args = self.cargs + self.targs
        cgate, _ = UnitaryDecomposition().execute(self.matrix)
        if len(mapping_args) == self.controls + self.targets:
            cgate & mapping_args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


CU3 = CU3Gate()


class FSimGate(BasicGate):
    """fSim gate"""

    def __init__(self, params: list = [np.pi / 2, 0]):
        super().__init__(
            controls=0,
            targets=2,
            params=2,
            type_=GateType.fsim,
            matrix_type=MatrixType.ctrl_normal,
        )

        self.pargs = params

    def __call__(self, alpha, beta):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate
            beta (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        params = [alpha, beta]
        for param in params:
            self.permit_element(param)

        return FSimGate(params)

    @property
    def matrix(self):
        costh = np.cos(self.pargs[0])
        sinth = np.sin(self.pargs[0])
        phi = self.pargs[1]

        return np.array(
            [
                [1, 0, 0, 0],
                [0, costh, -1j * sinth, 0],
                [0, -1j * sinth, costh, 0],
                [0, 0, 0, np.exp(-1j * phi)],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _FSim = self.copy()
        _FSim.pargs = [-self.pargs[0], -self.pargs[1]]

        return _FSim


FSim = FSimGate()


class RxxGate(BasicGate):
    """Rxx gate"""

    def __init__(self, params: list = [0]):
        super().__init__(
            controls=0,
            targets=2,
            params=1,
            type_=GateType.rxx,
            matrix_type=MatrixType.normal_normal,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return RxxGate([alpha])

    @property
    def matrix(self):
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)

        return np.array(
            [
                [costh, 0, 0, -1j * sinth],
                [0, costh, -1j * sinth, 0],
                [0, -1j * sinth, costh, 0],
                [-1j * sinth, 0, 0, costh],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _Rxx = self.copy()
        _Rxx.pargs = [-self.parg]

        return _Rxx

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            H & 0
            H & 1
            CX & [0, 1]
            Rz(self.parg) & 1
            CX & [0, 1]
            H & 0
            H & 1

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


Rxx = RxxGate()


class RyyGate(BasicGate):
    """Ryy gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=2,
            params=1,
            type_=GateType.ryy,
            matrix_type=MatrixType.normal_normal,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return RyyGate([alpha])

    @property
    def matrix(self):
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)

        return np.array(
            [
                [costh, 0, 0, 1j * sinth],
                [0, costh, -1j * sinth, 0],
                [0, -1j * sinth, costh, 0],
                [1j * sinth, 0, 0, costh],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _Ryy = self.copy()
        _Ryy.pargs = [-self.parg]

        return _Ryy

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            Hy & 0
            Hy & 1
            CX & [0, 1]
            Rz(self.parg) & 1
            CX & [0, 1]
            Hy & 0
            Hy & 1

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


Ryy = RyyGate()


class RzzGate(BasicGate):
    """Rzz gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=2,
            params=1,
            type_=GateType.rzz,
            matrix_type=MatrixType.diag_diag,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return RzzGate([alpha])

    @property
    def matrix(self):
        expth = np.exp(0.5j * self.parg)
        sexpth = np.exp(-0.5j * self.parg)

        return np.array(
            [[sexpth, 0, 0, 0], [0, expth, 0, 0], [0, 0, expth, 0], [0, 0, 0, sexpth]],
            dtype=self._precision,
        )

    def inverse(self):
        _Rzz = self.copy()
        _Rzz.pargs = [-self.parg]

        return _Rzz

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            CX & [0, 1]
            Rz(self.parg) & 1
            CX & [0, 1]

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


Rzz = RzzGate()


class RzxGate(BasicGate):
    """Rzx gate"""

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=2,
            params=1,
            type_=GateType.rzx,
            matrix_type=MatrixType.diag_normal,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return RzxGate([alpha])

    @property
    def matrix(self):
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)

        return np.array(
            [
                [costh, -1j * sinth, 0, 0],
                [-1j * sinth, costh, 0, 0],
                [0, 0, costh, 1j * sinth],
                [0, 0, 1j * sinth, costh],
            ],
            dtype=self._precision,
        )

    def inverse(self):
        _Rzx = self.copy()
        _Rzx.pargs = [-self.parg]

        return _Rzx

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            H & 0
            CX & [0, 1]
            Rz(self.parg) & 1
            CX & [0, 1]
            H & 0

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


Rzx = RzxGate()


class MeasureGate(BasicGate):
    """z-axis Measure gate

    Measure one qubit along z-axis.
    After acting on the qubit(circuit flush), the qubit get the value 0 or 1
    and the amplitude changed by the result.
    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.measure,
            matrix_type=MatrixType.special,
        )

    @property
    def matrix(self) -> np.ndarray:
        raise GateMatrixError("try to get the matrix of measure gate")


Measure = MeasureGate()


class ResetGate(BasicGate):
    """Reset gate

    Reset the qubit into 0 state,
    which change the amplitude
    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.reset,
            matrix_type=MatrixType.special,
        )

    @property
    def matrix(self) -> np.ndarray:
        raise GateMatrixError("try to get the matrix of reset gate")


Reset = ResetGate()


class BarrierGate(BasicGate):
    """Barrier gate

    In IBMQ, barrier gate forbid the optimization cross the gate,
    It is invalid in out circuit now.
    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type_=GateType.barrier,
            matrix_type=MatrixType.special,
        )

    @property
    def matrix(self) -> np.ndarray:
        raise GateMatrixError("try to get the matrix of barrier gate")


Barrier = BarrierGate()


class SwapGate(BasicGate):
    """Swap gate

    In the computation, it will not change the amplitude.
    Instead, it change the index of a Tangle.
    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=2,
            params=0,
            type_=GateType.swap,
            matrix_type=MatrixType.swap,
        )

        self.matrix = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            CX & [0, 1]
            CX & [1, 0]
            CX & [0, 1]

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


Swap = SwapGate()


class iSwapGate(BasicGate):
    """iSwap gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=2,
            params=0,
            type_=GateType.iswap,
            matrix_type=MatrixType.swap,
        )

        self.matrix = np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )


iSwap = iSwapGate()


class iSwapDaggerGate(BasicGate):
    """iSwap gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=2,
            params=0,
            type_=GateType.iswapdg,
            matrix_type=MatrixType.swap,
        )

        self.matrix = np.array(
            [[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )


iSwap_dagger = iSwapDaggerGate()


class SquareRootiSwapGate(BasicGate):
    """Square Root of iSwap gate"""

    def __init__(self):
        super().__init__(
            controls=0,
            targets=2,
            params=0,
            type_=GateType.sqiswap,
            matrix_type=MatrixType.swap,
        )

        self.matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, (1 + 1j) / np.sqrt(2), 0],
                [0, (1 + 1j) / np.sqrt(2), 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )


sqiSwap = SquareRootiSwapGate()


# PermGate class -- no qasm
class PermGate(BasicGate):
    """Permutation gate

    A special gate defined in our circuit,
    It can change an n-qubit qureg's amplitude by permutaion,
    the parameter is a 2^n list describes the permutation.
    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=0,
            params=0,
            type_=GateType.perm,
            matrix_type=MatrixType.special,
        )

    def __call__(self, targets: int, params: list):
        """pass permutation to the gate

        the length of params must be n, and should be a permutation for [0, n) without repeat

        Args:
            targets(int): the number of qubits
            params(list): the permutation parameters

        Returns:
            PermGate: the gate after filled by parameters
        """
        if not isinstance(params, list):
            raise TypeError("PermGate.params", "list", type(params))
        if not isinstance(targets, int):
            raise TypeError("PermGate.targets", "int", type(targets))

        assert len(params) == targets, GateParametersAssignedError(
            "the length of params must equal to targets"
        )

        _gate = self.copy()
        _gate.targets = targets
        _gate.params = targets
        for idx in params:
            if not isinstance(idx, int):
                raise TypeError("PermGate.params.values", "int", type(idx))
            if idx < 0 or idx >= _gate.targets:
                raise ValueError("PermGate.params.values", f"[0, {targets}]", idx)
            if idx in _gate.pargs:
                raise ValueError(
                    "PermGate.params.values", "have no duplicated value", idx
                )

            _gate.pargs.append(idx)

        return _gate

    def inverse(self):
        _gate = self.copy()
        _gate.pargs = [self.targets - 1 - p for p in self.pargs]

        return _gate

    def build_gate(self, targs: list = None):
        from QuICT.core.gate import CompositeGate

        swap_args: list[list[int]] = perm_decomposition(self.pargs[:])
        cgate = CompositeGate()
        with cgate:
            for swap_arg in swap_args:
                Swap & swap_arg

        if targs is not None:
            assert len(targs) == self.targets + self.controls, GateQubitAssignedError(
                "The qubits number should equal to the quantum gate."
            )
            cgate & targs

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


Perm = PermGate()


class PermFxGate(BasicGate):
    """act an Fx oracle on a qureg

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=0,
            params=0,
            type_=GateType.perm_fx,
            matrix_type=MatrixType.normal,
        )

    def __call__(self, n: int, params: list):
        """pass Fx to the gate

        Args:
            n (int): the number of targets
            params (list[int]): the list of index, and the index represent which should be 1.

        Returns:
            PermFxGate: the gate after filled by parameters
        """
        if not isinstance(params, list) or not isinstance(n, int):
            raise TypeError(
                f"n must be int {type(n)}, params must be list {type(params)}"
            )

        N = 1 << n
        for p in params:
            if p >= N:
                raise Exception("the params should be less than N")

        _gate = self.copy()
        _gate.params = 1 << (n + 1)
        _gate.targets = n + 1
        for idx in range(1 << _gate.targets):
            if idx >> 1 in params:
                _gate.pargs.append(idx ^ 1)
            else:
                _gate.pargs.append(idx)

        _gate.matrix = self._build_matrix(_gate.targets, _gate.pargs)

        return _gate

    def _build_matrix(self, targets, pargs):
        matrix_ = np.zeros((1 << targets, 1 << targets), dtype=self.precision)
        for idx, p in enumerate(pargs):
            matrix_[idx, p] = 1

        return matrix_


PermFx = PermFxGate()


class UnitaryGate(BasicGate):
    """Custom gate

    act an unitary matrix on the qureg,
    the parameters is the matrix

    """

    def __init__(self):
        super().__init__(controls=0, targets=0, params=0, type_=GateType.unitary)

    def __call__(self, params: np.array, matrix_type: MatrixType = MatrixType.normal):
        """pass the unitary matrix

        Args:
            params(np.array/list): contain 2^n * 2^n elements, which
            form an unitary matrix.

        Returns:
            UnitaryGateGate: the gate after filled by parameters
        """
        _u = UnitaryGate()

        if isinstance(params, list):
            params = np.array(params, dtype=self._precision)

        matrix_size = params.size
        if matrix_size == 0:
            raise GateMatrixError("the list or tuple passed in shouldn't be empty")

        length, width = params.shape
        if length != width:
            N = int(np.log2(matrix_size))
            assert N ^ 2 == matrix_size, GateMatrixError(
                "the shape of unitary matrix should be square."
            )

            params = params.reshape(N, N)

        n = int(np.log2(params.shape[0]))
        if (1 << n) != params.shape[0]:
            raise GateMatrixError(
                "the length of list should be the square of power(2, n)"
            )

        _u.targets = n
        _u.matrix = params.astype(self._precision)
        if n <= 3:
            _u._validate_matrix_type()
        else:
            _u._matrix_type = matrix_type

        return _u

    def _validate_matrix_type(self):
        if self._is_diagonal():
            is_control = np.allclose(
                self.matrix[:-1, :-1],
                np.identity((2**self.targets - 1), dtype=self._precision),
            )
            self._matrix_type = (
                MatrixType.control if is_control else MatrixType.diagonal
            )

        if (
            np.allclose(
                self.matrix[:-2, :-2],
                np.identity((2**self.targets - 2), dtype=self._precision),
            )
            and np.sum(self.matrix[:-2, -1]) + self.matrix[-1, -1] == 0
            and np.sum(self.matrix[-1, :-2]) + self.matrix[-2, -2] == 0
        ):
            self._matrix_type = MatrixType.reverse

    def copy(self):
        gate = super().copy()
        gate.matrix = self.matrix

        return gate

    def inverse(self):
        _U = super().copy()
        inverse_matrix = np.array(
            np.mat(self._matrix)
            .reshape(1 << self.targets, 1 << self.targets)
            .H.reshape(1, -1),
            dtype=self._precision,
        )
        _U.matrix = inverse_matrix
        _U.targets = self.targets

        return _U

    def build_gate(self):
        from QuICT.qcda.synthesis import UnitaryDecomposition

        assert self.controls + self.targets > 0
        mapping_args = self.cargs + self.targs
        cgate, _ = UnitaryDecomposition().execute(self.matrix)
        cgate & mapping_args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


Unitary = UnitaryGate()


class CCXGate(BasicGate):
    """Toffoli gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate

    """

    def __init__(self):
        super().__init__(
            controls=2,
            targets=1,
            params=0,
            type_=GateType.ccx,
            matrix_type=MatrixType.reverse,
        )

        self.matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ],
            dtype=self._precision,
        )

        self._target_matrix = np.array([[0, 1], [1, 0]], dtype=self._precision)

    @property
    def target_matrix(self) -> np.ndarray:
        return self._target_matrix

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            H & 2
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [0, 2]
            T_dagger & 2
            CX & [0, 2]
            T & 0
            T & 2
            H & 2

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


CCX = CCXGate()


class CCZGate(BasicGate):
    """Multi-control Z gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate

    """

    def __init__(self):
        super().__init__(controls=2, targets=1, params=0, type_=GateType.ccz)

        self.matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, -1],
            ],
            dtype=self._precision,
        )

        self._target_matrix = np.array([[1, 0], [0, -1]], dtype=self._precision)

    @property
    def target_matrix(self) -> np.ndarray:
        return self._target_matrix

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [0, 2]
            T_dagger & 2
            CX & [0, 2]
            T & 0
            T & 2

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


CCZ = CCZGate()


class CCRzGate(BasicGate):
    """controlled-Rz gate with two control bits"""

    def __init__(self, params: list = [0]):
        super().__init__(
            controls=2,
            targets=1,
            params=1,
            type_=GateType.ccrz,
            matrix_type=MatrixType.diagonal,
        )

        self.pargs = params

    def __call__(self, alpha):
        """Set parameters for the gate.

        Args:
            alpha (int/float/complex): The parameter for gate

        Raises:
            TypeError: param not one of int/float/complex

        Returns:
            BasicGate: The gate with parameters
        """
        self.permit_element(alpha)

        return CCRzGate([alpha])

    @property
    def matrix(self):
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, np.exp(-self.parg / 2 * 1j), 0],
                [0, 0, 0, 0, 0, 0, 0, np.exp(self.parg / 2 * 1j)],
            ],
            dtype=self._precision,
        )

    @property
    def target_matrix(self) -> np.ndarray:
        return np.array(
            [[np.exp(-self.parg / 2 * 1j), 0], [0, np.exp(self.parg / 2 * 1j)]],
            dtype=self._precision,
        )

    def inverse(self):
        _CCRz = self.copy()
        _CCRz.pargs = -self.parg
        return _CCRz

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            CRz(self.parg / 2) & [1, 2]
            CX & [0, 1]
            CRz(-self.parg / 2) & [1, 2]
            CX & [0, 1]
            CRz(self.parg / 2) & [0, 2]

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


CCRz = CCRzGate()


class QFTGate(BasicGate):
    """QFT gate"""

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            cgate = self.build_gate()
            self._matrix = cgate.matrix()
        return self._matrix

    def __init__(self, targets: int = 3):
        super().__init__(controls=0, targets=targets, params=0, type_=GateType.qft)

    def __call__(self, targets: int):
        """pass the unitary matrix

        Args:
            targets(int): point out the number of bits of the gate

        Returns:
            QFTGate: the QFTGate after filled by target number
        """
        return QFTGate(targets)

    def inverse(self):
        _IQFT = IQFTGate()
        _IQFT.targs = copy.deepcopy(self.targs)
        _IQFT.targets = self.targets
        return _IQFT

    def build_gate(self, targets: int = 0):
        from QuICT.core.gate import CompositeGate

        if targets == 0:
            targets = self.targets

        cgate = CompositeGate()
        with cgate:
            for i in range(targets):
                H & i
                for j in range(i + 1, targets):
                    CU1(2 * np.pi / (1 << j - i + 1)) & [j, i]

        args = self.cargs + self.targs
        if len(args) == targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


QFT = QFTGate()


class IQFTGate(QFTGate):
    """IQFT gate"""

    def __call__(self, targets: int):
        """pass the unitary matrix

        Args:
            targets(int): point out the number of bits of the gate

        Returns:
            IQFTGate: the IQFTGate after filled by target number
        """
        return IQFTGate(targets)

    def inverse(self):
        _QFT = QFTGate()
        _QFT.targs = copy.deepcopy(self.targs)
        _QFT.targets = self.targets
        return _QFT

    def build_gate(self, targets: int = 0):
        from QuICT.core.gate import CompositeGate

        if targets == 0:
            targets = self.targets

        cgate = CompositeGate()
        with cgate:
            for i in range(targets - 1, -1, -1):
                for j in range(targets - 1, i, -1):
                    CU1(-2 * np.pi / (1 << j - i + 1)) & [j, i]
                H & i

        args = self.cargs + self.targs
        if len(args) == targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


IQFT = IQFTGate()


class CSwapGate(BasicGate):
    """Fredkin gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate
    """

    def __init__(self):
        super().__init__(
            controls=1,
            targets=2,
            params=0,
            type_=GateType.cswap,
            matrix_type=MatrixType.swap,
        )

        self.matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=self._precision,
        )

        self._target_matrix = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=self._precision,
        )

    @property
    def target_matrix(self) -> np.ndarray:
        return self._target_matrix

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        with cgate:
            CX & [2, 1]
            H & 2
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [0, 2]
            T_dagger & 2
            CX & [0, 2]
            T & 0
            T & 2
            H & 2
            CX & [2, 1]

        args = self.cargs + self.targs
        if len(args) == self.controls + self.targets:
            cgate & args

        if self._precision == np.complex64:
            cgate.convert_precision()

        return cgate


CSwap = CSwapGate()
=======
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
Hy = BasicGate(*GATEINFO_MAP[GateType.hy], is_original_gate=True)
S = BasicGate(*GATEINFO_MAP[GateType.s], is_original_gate=True)
S_dagger = BasicGate(*GATEINFO_MAP[GateType.sdg], is_original_gate=True)
X = BasicGate(*GATEINFO_MAP[GateType.x], is_original_gate=True)
Y = BasicGate(*GATEINFO_MAP[GateType.y], is_original_gate=True)
Z = BasicGate(*GATEINFO_MAP[GateType.z], is_original_gate=True)
SX = BasicGate(*GATEINFO_MAP[GateType.sx], is_original_gate=True)
SY = BasicGate(*GATEINFO_MAP[GateType.sy], is_original_gate=True)
SW = BasicGate(*GATEINFO_MAP[GateType.sw], is_original_gate=True)
ID = BasicGate(*GATEINFO_MAP[GateType.id], is_original_gate=True)
U1 = BasicGate(*GATEINFO_MAP[GateType.u1], is_original_gate=True)
U2 = BasicGate(*GATEINFO_MAP[GateType.u2], is_original_gate=True)
U3 = BasicGate(*GATEINFO_MAP[GateType.u3], is_original_gate=True)
Rx = BasicGate(*GATEINFO_MAP[GateType.rx], is_original_gate=True)
Ry = BasicGate(*GATEINFO_MAP[GateType.ry], is_original_gate=True)
Rz = BasicGate(*GATEINFO_MAP[GateType.rz], is_original_gate=True)
T = BasicGate(*GATEINFO_MAP[GateType.t], is_original_gate=True)
T_dagger = BasicGate(*GATEINFO_MAP[GateType.tdg], is_original_gate=True)
Phase = BasicGate(*GATEINFO_MAP[GateType.phase], is_original_gate=True)
GPhase = BasicGate(*GATEINFO_MAP[GateType.gphase], is_original_gate=True)
CZ = BasicGate(*GATEINFO_MAP[GateType.cz], is_original_gate=True)
CX = BasicGate(*GATEINFO_MAP[GateType.cx], is_original_gate=True)
CY = BasicGate(*GATEINFO_MAP[GateType.cy], is_original_gate=True)
CH = BasicGate(*GATEINFO_MAP[GateType.ch], is_original_gate=True)
CRy = BasicGate(*GATEINFO_MAP[GateType.cry], is_original_gate=True)
CRz = BasicGate(*GATEINFO_MAP[GateType.crz], is_original_gate=True)
CU1 = BasicGate(*GATEINFO_MAP[GateType.cu1], is_original_gate=True)
CU3 = BasicGate(*GATEINFO_MAP[GateType.cu3], is_original_gate=True)
FSim = BasicGate(*GATEINFO_MAP[GateType.fsim], is_original_gate=True)
Rxx = BasicGate(*GATEINFO_MAP[GateType.rxx], is_original_gate=True)
Ryy = BasicGate(*GATEINFO_MAP[GateType.ryy], is_original_gate=True)
Rzz = BasicGate(*GATEINFO_MAP[GateType.rzz], is_original_gate=True)
Rzx = BasicGate(*GATEINFO_MAP[GateType.rzx], is_original_gate=True)
Measure = BasicGate(*GATEINFO_MAP[GateType.measure], is_original_gate=True)
Reset = BasicGate(*GATEINFO_MAP[GateType.reset], is_original_gate=True)
Barrier = BasicGate(*GATEINFO_MAP[GateType.barrier], is_original_gate=True)
Swap = BasicGate(*GATEINFO_MAP[GateType.swap], is_original_gate=True)
iSwap = BasicGate(*GATEINFO_MAP[GateType.iswap], is_original_gate=True)
iSwap_dagger = BasicGate(*GATEINFO_MAP[GateType.iswapdg], is_original_gate=True)
sqiSwap = BasicGate(*GATEINFO_MAP[GateType.sqiswap], is_original_gate=True)
CCX = BasicGate(*GATEINFO_MAP[GateType.ccx], is_original_gate=True)
CCZ = BasicGate(*GATEINFO_MAP[GateType.ccz], is_original_gate=True)
CCRz = BasicGate(*GATEINFO_MAP[GateType.ccrz], is_original_gate=True)
CSwap = BasicGate(*GATEINFO_MAP[GateType.cswap], is_original_gate=True)
>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
