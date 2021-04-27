"""
Class for customizing the process of synthesis, optimization and mapping
"""

import numpy as np

from QuICT.core import Circuit, CompositeGate
from .mapping._mapping import Mapping
from .optimization._optimization import Optimization
from .synthesis._synthesis import Synthesis

class QCDA(object):
    """ Customize the process of synthesis, optimization and mapping

    In this class, we provide the users with a direct path to design the process
    by which they could transform a unitary matrix to a quantum circuit and/or
    optimize a quantum circuit.

    Users could choose whether to use the synthesis, optimization and mapping
    operations implemented by us or to develop their own operations, only if
    the executions are similar to the original ones.
    """
    def __init__(self, process, config):
        """ Initialize a QCDA process and its configuration

        Args:
            process(list): A list of synthesis, optimization and mapping operations
            config(dict): A dictionary which saves the configuration of the QCDA
                process for the convenience of controlling the process
        
        Note:
            In current version, config is only used to store the parameters to be passed
            to the operations. The key to the parameters of a certain operation must be
            operation.__name__, otherwise the parameters would not be passed.
        """
        self.process = process
        self.config = config

    def execute(self, gates):
        """ Execute the QCDA Process

        By default, the operations in the self.process would be executed in sequence,
        with the configuration saved in self.config. The output of the previous operation
        would be the input of the current one, so be aware that the type of input and
        output must be aligned.

        Args:
            gates(np.ndarray/CompositeGate/Circuit): The unitary matrix, CompositeGate
                or circuit to be transformed

        Returns:
            Circuit: The circuit derived from the QCDA process
        """
        for operation in self.process:
            # Pass the parameter to operation
            try:
                config = self.config[operation.__name__]
                operation = operation(config)
            except KeyError:
                pass

            if isinstance(operation, Synthesis):
                gates = self._execute_synthesis(operation, gates)
            if isinstance(operation, Optimization):
                gates = self._execute_optimization(operation, gates)
            if isinstance(operation, Mapping):
                circuit = self._execute_mapping(operation, gates)

            return circuit

    def _execute_synthesis(self, operation, gates):
        """ Execute Synthesis class

        As is the expected first steps of QCDA, Synthesis would face the most complicated
        conditions. It would deal with different input as follows:
        1. If the gates is a unitary matrix, it will construct a CompositeGate equivalent
        to the matrix.
        2. If the gates is a CompositeGate, it will check and transform the gates in the 
        CompositeGate.
        3. If the gates is a Circuit, it will extract the gates in the circuit as a
        CompositeGate and work as case 2.

        Args:
            operation(Synthesis): Operation to be executed
            gates(np.ndarray/CompositeGate/Circuit): Object to be transformed
        
        Returns:
            CompositeGate: Transformation result
        """
        if isinstance(gates, np.ndarray):
            qubits = int(np.log2(gates.shape[0]))
            assert gates.ndim == 2 \
                and gates.shape[0] == gates.shape[1] \
                and 2 ** qubits == gates.shape[0], \
                ValueError("Matrix with wrong shape encountered")
            gates = operation.execute(gates)
            return gates

        if isinstance(gates, Circuit):
            gates = CompositeGate(gates)
        
        assert isinstance(gates, CompositeGate), \
            TypeError("Invalid type of input for Synthesis execution.")
        gates = operation.execute(gates)
        return gates

    def _execute_optimization(self, operation, gates):
        """ Execute Optimization process

        Args:
            operation(Optimization): Operation to be executed
            gates(CompositeGate/Circuit): Object to be transformed

        Returns:
            CompositeGate: Transformation result
        """
        if isinstance(gates, Circuit):
            gates = CompositeGate(gates)
        
        assert isinstance(gates, CompositeGate), \
            TypeError("Invalid type of input for Optimization execution.")
        gates = operation.execute(gates)
        return gates

    def _execute_mapping(self, operation, gates):
        """ Execute the Mapping process

        Args:
            operation(Mapping): Operation to be executed
            gates(CompositeGate/Circuit): Object to be transformed

        Returns:
            Circuit: Transformation result
        """
        # TODO: Align the interface 
        # FIXME: the input of Mapping.execute must be a circuit
        circuit = operation.execute(gates)
        return circuit
