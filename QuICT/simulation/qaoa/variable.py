import numpy as np
import parser


class Operator:
    def __init__(self):
        self._function = None

    def add_operator(self, variable, operator):
        pass

    def calculate(self, variable):
        pass


class Expression:
    def __init__(self, qubits: int):
        self._indexes = 1 << qubits
        self._operators = {}

    def add_operator(self, indexes, variable, operation):
        pass

    def calculate(self, variables):
        pass