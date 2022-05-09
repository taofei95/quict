import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate


class OutputAligner(object):
    """
    Decorating class that keeps the type of output aligned with that of input for QCDA execute functions

    Valid type of output are restricted to be CompositeGate or Circuit.
    For functions with other types of input or other special usages, the type of output could be assigned.
    """
    def __init__(self, output=None):
        """
        Args:
            output(class, optional): assigned type of output in [CompositeGate, Circuit]
        """
        if output is not None:
            assert output in [CompositeGate, Circuit], ValueError('Invalid output type')
        self.output = output

    def __call__(self, func):
        """
        Args:
            func(callable): function to be decorated

        Returns:
            callable: func with whose output type aligned
        """
        def align_func(object, input):
            """
            For QCDA execute functions, their input would be restricted to be one argument representing
            the input that would be synthesized, optimized or mapped, while other parameters needed in
            the function would be passed by the __init__ method.

            Args:
                object(Object): the object that func belongs to
                input: input of func, the following types are supported.
                    1. CompositeGate: the input is a CompositeGate
                    2. Circuit: the input is a Circuit
                    3. other: any other reasonable input

            Returns:
                CompositeGate/Circuit: output of func with whose output type aligned
            """
            if self.output is None:
                self.output = type(input)
            assert self.output in [CompositeGate, Circuit], TypeError('Invalid output type')

            # Record width of input
            width = None
            if isinstance(input, CompositeGate) or isinstance(input, Circuit):
                width = input.width()

            # Execute the function
            output = func(object, input)

            # Align the output
            if isinstance(output, self.output):
                return output
            if self.output == CompositeGate:
                output: Circuit
                gates = CompositeGate(gates=output.gates)
                return gates
            if self.output == Circuit:
                output: CompositeGate
                if width is None:
                    width = output.width()
                circuit = Circuit(width)
                circuit.extend(output)
                return circuit

        return align_func
