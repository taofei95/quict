from QuICT.tools.exception import QuICTException


class HamiltonianError(QuICTException):
    """Hamiltonian Error."""

    def __init__(self, msg: str = None):
        super().__init__(3006, msg)


class VQEModelError(QuICTException):
    """Hamiltonian Error."""

    def __init__(self, msg: str = None):
        super().__init__(3007, msg)


class QNNModelError(QuICTException):
    """Hamiltonian Error."""

    def __init__(self, msg: str = None):
        super().__init__(3008, msg)
