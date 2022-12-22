"""
Class for customizing the whole process of synthesis, optimization and mapping
"""

from QuICT.core.utils import GateType
from QuICT.core.utils.circuit_info import CircuitMode
from QuICT.qcda.mapping import MCTSMapping
from QuICT.qcda.optimization import (CliffordRzOptimization,
                                     CommutativeOptimization,
                                     SymbolicCliffordOptimization,
                                     TemplateOptimization)
from QuICT.qcda.optimization.circuit_partition.circuit_partition_optimization import \
    CircuitPartitionOptimization
from QuICT.qcda.synthesis import GateDecomposition, GateTransform


class QCDA(object):
    """ Customize the process of synthesis, optimization and mapping

    In this class, we meant to provide the users with a direct path to design the
    process by which they could transform a unitary matrix to a quantum circuit
    and/or optimize a quantum circuit.
    """

    def __init__(self, process=None):
        """ Initialize a QCDA process

        A QCDA process is defined by a list of synthesis, optimization and mapping.
        Experienced users could customize the process for certain purposes.

        Args:
            instruction(InstructionSet, optional): InstructionSet for default process
            layout(Layout, optional): Layout for default process
            process(list, optional): A customized list of Synthesis, Optimization and Mapping
        """
        self.process = []
        if process is not None:
            self.process = process

    def add_method(self, method=None):
        """ Adding a specific method to the process

        Args:
            method: Some QCDA method
        """
        self.process.append(method)

    def add_default_synthesis(self, target_instruction=None):
        """ Generate the default synthesis process

        The default synthesis process contains the GateDecomposition and GateTransform, which would
        transform the gates in the original Circuit/CompositeGate to a certain InstructionSet.

        Args:
            instruction(InstructionSet): The target InstructionSet
        """
        assert target_instruction is not None, ValueError('No InstructionSet provided for Synthesis')
        self.add_method(GateDecomposition())
        self.add_method(GateTransform(target_instruction))

    def add_default_optimization(self, level='light'):
        """ Generate the default optimization process

        The default optimization process contains the CommutativeOptimization.

        Args:
            level(str): Optimizing level. Support `light`, `heavy` level.
            cost(str): Cost measure. Support `nisq` and `fault_tolerant`.
        """

        if level == 'light':
            opt = CircuitPartitionOptimization()
            opt.add_optimizer(CircuitMode.Clifford, SymbolicCliffordOptimization())
            # opt.add_optimizer(CircuitMode.CliffordRz, CommutativeOptimization())
            opt.add_optimizer(CircuitMode.CliffordRz, CliffordRzOptimization(level='light'))
            opt.add_optimizer(
                CircuitMode.Arithmetic,
                TemplateOptimization(template_typelist=[GateType.x, GateType.cx, GateType.ccx])
            )
            opt.add_optimizer(CircuitMode.Misc, CommutativeOptimization())
            self.add_method(opt)

        elif level == 'heavy':
            opt = CircuitPartitionOptimization()

            opt.add_optimizer(CircuitMode.Clifford, SymbolicCliffordOptimization())
            opt.add_optimizer(CircuitMode.Clifford, CliffordRzOptimization(level='light'))
            opt.add_optimizer(CircuitMode.CliffordRz, CliffordRzOptimization(level='heavy'))
            opt.add_optimizer(
                CircuitMode.Arithmetic,
                TemplateOptimization(template_typelist=[GateType.x, GateType.cx, GateType.ccx]),
            )
            opt.add_optimizer(CircuitMode.Arithmetic, CliffordRzOptimization(level='light'))

            opt.add_optimizer(CircuitMode.Misc, CommutativeOptimization())
            self.add_method(opt)

    def add_default_mapping(self, layout=None):
        """ Generate the default mapping process

        The default mapping process contains the Mapping

        Args:
            layout(Layout): Topology of the target physical device
        """
        assert layout is not None, ValueError('No Layout provided for Mapping')
        self.add_method(MCTSMapping(layout, init_mapping_method='anneal'))

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
