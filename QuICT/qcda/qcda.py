"""
Class for customizing the process of synthesis, optimization and mapping
"""

import numpy as np

from QuICT.core import Circuit, CompositeGate
from QuICT.qcda.synthesis.unitary_transform import UnitaryTransform
from QuICT.tools.interface import OPENQASMInterface
from .synthesis._synthesis import Synthesis
from .optimization._optimization import Optimization
from .mapping._mapping import Mapping
from .synthesis import GateDecomposition, GateTransform
from .optimization import CommutativeOptimization
from .mapping import MCTSMapping


class QCDA(object):
    """ Customize the process of synthesis, optimization and mapping

    In this class, we meant to provide the users with a direct path to design the
    process by which they could transform a unitary matrix to a quantum circuit
    and/or optimize a quantum circuit.

    In this version, users could choose whether to use the synthesis, optimization
    and mapping operations implemented by us.

    XXX: Although the structure of `process` is kept, now the revision of `process`
    is highly restricted in case of unexpected behaviour resulting from inappropriate
    modification. Now the users could only overload the `compile` function to completely
    control the workflow of `process`. Still, it is needed that a certain design for
    allowing the users to revise the `process` with their own operations, only if the
    executions are similar to the original ones.(How to ensure that the user gives a proper
    `process`? That's why it is not implemented in this version.)

    TODO: Optimization before synthesis is a good idea. However, because of the lack
    of methods to deal with complex gates, this subprocess is omitted in this version.

    TODO: Some optimization processes are related to topology, the structure of the process
    might be revised.
    """
    def __init__(self):
        """ Initialize a QCDA process

        Args:
            process(list): A list of synthesis, optimization and mapping operations

        Note:
            The element of `process` must be a list formatted as [operation, args, kwargs],
            in which `operation`, `args`, `kwargs` are the class of operation, the arguments
            and the keyword arguments respectively.
        """
        self.process = []

    @classmethod
    def load_gates(cls, objective):
        """ Load the objective to CompositeGate with BasicGates only

        The objective would be analyzed and tranformed to a CompositeGate depending on its
        type, with the ComplexGates in it decomposed to BasicGates.

        Args:
            objective: objective of QCDA process, the following types are supported.
                1. str: the objective is the path of an OPENQASM file
                2. numpy.ndarray: the objective is a unitary matrix
                3. Circuit: the objective is a Circuit
                4. CompositeGate: the objective is a CompositeGate

        Returns:
            CompositeGate: gates equivalent to the objective, with BasicGates only

        Raises:
            If the objective could not be resolved as any of the above types.
        """
        # Load the objective as gates
        if isinstance(objective, np.ndarray):
            gates, _ = UnitaryTransform.execute(objective)

        if isinstance(objective, str):
            qasm = OPENQASMInterface.load_file(objective)
            if qasm.valid_circuit:
                # FIXME: no circuit here
                circuit = qasm.circuit
                gates = CompositeGate(circuit)
            else:
                raise ValueError("Invalid qasm file!")

        if isinstance(objective, Circuit):
            gates = CompositeGate(objective)

        if isinstance(objective, CompositeGate):
            gates = CompositeGate(objective)

        assert isinstance(gates, CompositeGate), TypeError('Invalid objective!')
        return gates

    @staticmethod
    def default_synthesis(instruction):
        """ Generate the default synthesis process

        The default synthesis process contains the GateDecomposition and GateTransform, which would
        transform the gates in the original Circuit/CompositeGate to a certain InstructionSet.

        Args:
            instruction(InstructionSet): The target InstructionSet

        Returns:
            List: Synthesis subprocess
        """
        assert instruction is not None,\
            ValueError('No InstructionSet provided for Synthesis')
        subprocess = []
        subprocess.append([GateDecomposition, [], {}])
        subprocess.append([GateTransform, [instruction], {}])
        return subprocess

    @staticmethod
    def default_optimization():
        """ Generate the default optimization process

        The default optimization process contains the CommutativeOptimization.
        TODO: Now TemplateOptimization only works for Clifford+T circuits, to be added.

        Returns:
            List: Optimization subprocess
        """
        subprocess = []
        subprocess.append([CommutativeOptimization, [], {}])
        return subprocess

    @staticmethod
    def default_mapping(layout):
        """ Generate the default mapping process

        The default mapping process contains the Mapping

        Args:
            layout(Layout): Topology of the target physical device

        Returns:
            List: Mapping subprocess
        """
        assert layout is not None,\
            ValueError('No Layout provided for Mapping')
        subprocess = []
        subprocess.append([MCTSMapping, [layout], {'init_mapping_method': 'anneal'}])
        return subprocess

    def compile(self, objective, instruction=None, layout=None, synthesis=True, optimization=True, mapping=True):
        """ Compile the objective with default process setting

        The easy-to-use process for the users to compile the objective with certain InstructionSet
        and topology. Three switches are given to control whether to use the corresponding subprocess.
        Be aware that synthesis subprocess needs `instruction` and mapping subprocess needs `topology`.

        Args:
            objective: objective of QCDA process, the following types are supported.
                1. str: the objective is the path of an OPENQASM file
                2. numpy.ndarray: the objective is a unitary matrix
                3. Circuit: the objective is a Circuit
                4. CompositeGate: the objective is a CompositeGate
            instruction(InstructionSet): The target InstructionSet
            layout(Layout): Topology of the target physical device
            synthesis(bool): whether to use synthesis subprocess(`instruction` needed)
            optimization(bool): whether to use optimization subprocess
            mapping(bool): whether to use mapping subprocess(`topology` needed)

        Returns:
            CompositeGate/Circuit: the resulting CompositeGate or Circuit(depending on whether to use
            the mapping process)
        """
        gates = self.load_gates(objective)

        if synthesis:
            self.process.extend(self.default_synthesis(instruction))
        if optimization:
            self.process.extend(self.default_optimization())
        if mapping:
            self.process.extend(self.default_mapping(layout))

        gates = self.__custom_compile(gates)
        self.process = []

        return gates

    def __custom_compile(self, gates):
        """ The execution of the QCDA process

        This is the exection part of the QCDA process, after the process is generated by default
        or created by advanced users.

        Args:
            gates(CompositeGate): The CompositeGate to be compiled by `self.process`

        Returns:
            CompositeGate/Circuit: the resulting CompositeGate or Circuit

        HACK: Feel free to hack this part if needed, though it works for default process.
        """
        for operation, args, kwargs in self.process:
            if isinstance(operation, Synthesis) or isinstance(operation, Optimization):
                gates = CompositeGate(gates)
            if isinstance(operation, Mapping):
                gates = gates.build_circuit()
            print('Processing {}'.format(operation.__name__))
            gates = operation.execute(gates, *args, **kwargs)
            print('Process {} finished'.format(operation.__name__))

        return gates
