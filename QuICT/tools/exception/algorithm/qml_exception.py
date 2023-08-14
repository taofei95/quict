from QuICT.tools.exception import QuICTException


class HamiltonianError(QuICTException):
    """Hamiltonian Error."""

    def __init__(self, msg: str = None):
        super().__init__(3001, msg)


class AnsatzShapeError(QuICTException):
    """Ansatz Shape Error."""

    def __init__(self, require: str, given: str):
        msg = f"The shape of the input parameters needs to match the defined ansatz, \
            which should be {require}, but given {given}"
        super().__init__(3002, msg)


class AnsatzValueError(QuICTException):
    """Ansatz Value Error."""

    def __init__(self, require: str, given: str):
        msg = f"The ansatz only supports the following gates: {require}, but given {given}."
        super().__init__(3003, msg)


class EncodingError(QuICTException):
    """Encoding Error."""

    def __init__(self, msg: str = None):
        super().__init__(3004, msg)


class ModelError(QuICTException):
    """Model Error."""

    def __init__(self, msg: str = None):
        super().__init__(3015, msg)


class DatasetError(QuICTException):
    """Dataset Error."""

    def __init__(self, msg: str = None):
        super().__init__(3016, msg)


class ModelRestoreError(QuICTException):
    """Model Restore Operator Error."""

    def __init__(self, msg: str = None):
        super().__init__(3007, msg)
