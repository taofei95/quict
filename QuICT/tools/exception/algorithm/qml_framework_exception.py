from QuICT.tools.exception import QuICTException


class AnsatzAppendError(QuICTException):
    """Ansatz Append Operator Error."""

    def __init__(self, msg: str = None):
        super().__init__(3001, msg)


class AnsatzForwardError(QuICTException):
    """Ansatz Forward Operator Error."""

    def __init__(self, msg: str = None):
        super().__init__(3002, msg)
