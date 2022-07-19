"""
Class for customizing the whole process of synthesis, optimization and mapping
"""

from .synthesis.synthesis import Synthesis
from .optimization.optimization import Optimization
from .mapping.mapping import Mapping


class QCDA(object):
    """ Customize the process of synthesis, optimization and mapping

    In this class, we meant to provide the users with a direct path to design the
    process by which they could transform a unitary matrix to a quantum circuit
    and/or optimize a quantum circuit.
    """
    def __init__(self, instruction=None, layout=None, process=None):
        """ Initialize a QCDA process

        A QCDA process is defined by a list of Synthesis, Optimization and Mapping.
        Experienced users could customize the process for certain purposes.
        Otherwise, passing optional InstructionSet and/or Layout would generate a default process.

        Args:
            instruction(InstructionSet, optional): InstructionSet for default process
            layout(Layout, optional): Layout for default process
            process(list, optional): A customized list of Synthesis, Optimization and Mapping
        """
        if process is not None:
            self.process = process
        else:
            self.process = []
            # Synthesis
            if instruction is not None:
                synthesis = Synthesis(instruction)
                self.process.append(synthesis)
            # Optimization
            optimization = Optimization()
            self.process.append(optimization)
            # Mapping
            if layout is not None:
                mapping = Mapping(layout)
                self.process.append(mapping)
            # Synthesis to decompose the swap gates
            if instruction is not None:
                self.process.append(synthesis)

    def compile(self, circuit):
        """ Compile the circuit with the given process

        Args:
            circuit(CompositeGate/Circuit): the target CompositeGate or Circuit

        Returns:
            CompositeGate/Circuit: the resulting CompositeGate or Circuit
        """
        for process in self.process:
            circuit = process.execute(circuit)

        return circuit
