import numpy as np
from typing import List
import inspect

from QuICT.core import *
from QuICT.qcda.optimization._optimization import Optimization
from dag import DAG


class AutoOptimization(Optimization):
    """
    Heuristic optimization of circuits in Clifford + Rz.

    [1] Nam, Yunseong, et al. "Automated optimization of large quantum
    circuits with continuous parameters." npj Quantum Information 4.1
    (2018): 1-12.
    """

    _optimize_sub_method = {
        1: "reduce_hadamard_gates",
        2: "cancel_single_qubit_gates",
        3: "cancel_two_qubit_gates",
        4: "merge_rotations",
        5: "float_rotations",
    }
    _optimize_routine = {
        'heavy': [1, 3, 2, 3, 1, 2, 5],
        'light': [1, 3, 2, 3, 1, 2, 4, 3, 2],
    }

    @classmethod
    def reduce_hadamard_gates(cls, gates: DAG):
        """
        how to design a template? a template need to have:
        1. a circuit equation
        2. a replacing method
        """
        pass

    @classmethod
    def cancel_single_qubit_gates(cls, gates: DAG):
        """
        1. iterate over all single qubit gate
        2. starting from a single qubit gate, search for commuting patterns
        """

    @classmethod
    def cancel_two_qubit_gates(cls, gates: DAG):
        print(inspect.currentframe(), 'not implemented yet')

    @classmethod
    def merge_rotations(cls, gates: DAG):
        print(inspect.currentframe(), 'not implemented yet')

    @classmethod
    def float_rotations(cls, gates: DAG):
        print(inspect.currentframe(), 'not implemented yet')

    @classmethod
    def _execute(cls, gates, routine: List[int]):
        _gates = DAG(CompositeGate(gates))
        while True:
            for step in routine:
                getattr(cls, cls._optimize_sub_method[step])(_gates)
            if True:
                print('break condition not added yet')
                break

        return _gates.get_circuit()

    @classmethod
    def execute(cls, gates, mode='light'):
        """
        Heuristic optimization of circuits in Clifford + Rz.

        Args:
              gates(Union[Circuit, CompositeGate]): Circuit to be optimized
              mode(str): Support 'light' and 'heavy' mode. See details in [1].
        Returns:
            CompositeGate: The CompositeGate after optimization

        [1] Nam, Yunseong, et al. "Automated optimization of large quantum
        circuits with continuous parameters." npj Quantum Information 4.1
        (2018): 1-12.
        """
        if mode in cls._optimize_routine:
            return cls._execute(gates, cls._optimize_routine[mode])
        else:
            raise Exception(f'unrecognized mode {mode}')
