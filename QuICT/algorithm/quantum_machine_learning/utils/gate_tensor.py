import copy
from typing import Union
import numpy as np
import torch

from QuICT.core.utils import (
    GateType, MatrixType, SPECIAL_GATE_SET, DIAGONAL_GATE_SET, CGATE_LIST,
    PAULI_GATE_SET, CLIFFORD_GATE_SET,
    perm_decomposition, matrix_product_to_circuit
)


class BasicGateTensor(object):
    """ the abstract SuperClass of all basic quantum gate

    All basic quantum gate described in the framework have
    some common attributes and some common functions
    which defined in this class

    Attributes:
        name(str): the name of the gate
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
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def matrix(self) -> torch.tensor:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix) -> torch.tensor:
        self._matrix = matrix

    @property
    def target_matrix(self) -> torch.tensor:
        return self.matrix

    @property
    def type(self):
        return self._type

    @property
    def matrix_type(self):
        return self._matrix_type

    @property
    def controls(self) -> int:
        return self._controls

    @controls.setter
    def controls(self, controls: int):
        assert isinstance(controls, int)
        self._controls = controls

    @property
    def cargs(self):
        return self._cargs

    @cargs.setter
    def cargs(self, cargs: Union[list, int]):
        if isinstance(cargs, int):
            cargs = [cargs]

        assert len(cargs) == len(set(cargs)), "Duplicated control qubit indexes."
        self._cargs = cargs

    @property
    def targets(self) -> int:
        return self._targets

    @targets.setter
    def targets(self, targets: int):
        assert isinstance(targets, int)
        self._targets = targets

    @property
    def targs(self):
        return self._targs

    @targs.setter
    def targs(self, targs: list):
        if isinstance(targs, int):
            targs = [targs]

        assert len(targs) == len(set(targs)), "Duplicated target qubit indexes."
        assert not set(self._cargs) & set(targs), "Same qubit indexes in control and target."
        self._targs = targs

    @property
    def params(self) -> int:
        return self._params

    @params.setter
    def params(self, params: int):
        self._params = params

    @property
    def pargs(self):
        return self._pargs

    @pargs.setter
    def pargs(self, pargs: list):
        if isinstance(pargs, list):
            self._pargs = pargs
        else:
            self._pargs = [pargs]

        assert len(self._pargs) == self.params

    @property
    def parg(self):
        return self.pargs[0]

    @property
    def carg(self):
        return self.cargs[0]

    @property
    def targ(self):
        return self.targs[0]

    @property
    def precision(self):
        return self._precision

    @property
    def qasm_name(self):
        return self._qasm_name

    def __init__(
        self,
        controls: int,
        targets: int,
        params: int,
        type: GateType,
        matrix_type: MatrixType = MatrixType.normal
    ):
        self._matrix = None

        self._controls = controls
        self._targets = targets
        self._params = params
        self._cargs = []    # list of int
        self._targs = []    # list of int
        self._pargs = []    # list of float/..

        assert isinstance(type, GateType)
        self._type = type
        self._matrix_type = matrix_type
        self._precision = np.complex128
        self._qasm_name = str(type.name)
        self._name = "-".join([str(type), "", ""])

        self.assigned_qubits = []   # list of qubits

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
        assert isinstance(other, BasicGate)
        if (
            self.type != other.type or
            (self.cargs + self.targs) != (other.cargs + other.targs) or
            not np.allclose(self.matrix, other.matrix)
        ):
            return False

        return True

    def update_name(self, qubit_id: str, circuit_idx: int = None):
        """ Updated gate's name with the given information

        Args:
            qubit_id (str): The qubit's unique ID.
            circuit_idx (int, optional): The gate's order index in the circuit. Defaults to None.
        """
        qubit_id = qubit_id[:6]
        name_parts = self.name.split('-')
        name_parts[1] = qubit_id

        if circuit_idx is not None:
            name_parts[2] = str(circuit_idx)

        self.name = '-'.join(name_parts)

    def __str__(self):
        """ get gate information """
        gate_info = {
            "name": self.name,
            "controls": self.controls,
            "control_bit": self.cargs,
            "targets": self.targets,
            "target_bit": self.targs,
            "parameters": self.pargs
        }

        return str(gate_info)

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

        gate.pargs = copy.deepcopy(self.pargs)
        gate.targs = copy.deepcopy(self.targs)
        gate.cargs = copy.deepcopy(self.cargs)

        if self.assigned_qubits:
            gate.assigned_qubits = copy.deepcopy(self.assigned_qubits)
            gate.update_name(gate.assigned_qubits[0].id)

        if self.precision == np.complex64:
            gate.convert_precision()

        return gate

    @staticmethod
    def permit_element(element):
        """ judge whether the type of a parameter is int/float/complex

        for a quantum gate, the parameter should be int/float/complex

        Args:
            element: the element to be judged

        Returns:
            bool: True if the type of element is int/float/complex
        """
        if isinstance(element, int) or isinstance(element, float) or isinstance(element, complex):
            return True
        else:
            tp = type(element)
            if tp == np.int64 or tp == np.float64 or tp == np.complex128:
                return True
            return False
