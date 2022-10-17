from QuICT.tools.exception import QuICTException


class CircuitTypeError(QuICTException):
    def __init__(self, msg: str = None):
        super().__init__(1001, msg)


class CircuitQubitsError(QuICTException):
    def __init__(self, msg: str = None):
        super().__init__(1002, msg)


class CircuitFidelityError(QuICTException):
    def __init__(self, msg: str = None):
        super().__init__(1003, msg)


class CircuitAppendError(QuICTException):
    def __init__(self, msg: str = None):
        super().__init__(1004, msg)


class CircuitReplaceError(QuICTException):
    def __init__(self, msg: str = None):
        super().__init__(1005, msg)


class CircuitSpecialAppendError(QuICTException):
    def __init__(self, msg: str = None):
        super().__init__(1006, msg)


class CircuitDAGError(QuICTException):
    def __init__(self, msg: str = None):
        super().__init__(1007, msg)


class CircuitDrawError(QuICTException):
    def __init__(self, msg: str = None):
        super().__init__(1008, msg)
