"""
Class for customizing the whole process of synthesis, optimization and mapping
"""
from QuICT.core import Circuit
from QuICT.core.virtual_machine import VirtualQuantumMachine
from QuICT.core.utils import GateType, CLIFFORD_GATE_SET
from QuICT.qcda.synthesis import GateTransform
from QuICT.qcda.optimization import (
    CommutativeOptimization, CliffordRzOptimization,
    CnotWithoutAncilla, SymbolicCliffordOptimization
)
from QuICT.qcda.mapping import MCTSMapping, SABREMapping
from QuICT.tools import Logger


logger = Logger("QCDA")


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

    def add_gate_transform(self, target_instruction=None):
        """ Add GateTransform for some target InstructionSet

        GateTransform would transform the gates in the original Circuit/CompositeGate to a certain InstructionSet.

        Args:
            target_instruction(InstructionSet): The target InstructionSet
        """
        assert target_instruction is not None, ValueError('No InstructionSet provided for Synthesis')
        self.add_method(GateTransform(target_instruction))

    def add_default_optimization(self, level='light', keep_phase=False):
        """ Generate the default optimization process

        The default optimization process contains the CommutativeOptimization.

        Args:
            level(str): Optimizing level. Support `light`, `heavy` level.
            keep_phase(bool): whether to keep the global phase as a GPhase gate in the output
        """

        self.add_method(CommutativeOptimization(keep_phase=keep_phase))
        self.add_method(CliffordRzOptimization(level=level, keep_phase=keep_phase))

    def add_mapping(self, layout=None, method='sabre'):
        """ Generate the default mapping process

        The default mapping process contains the Mapping

        Args:
            layout(Layout): Topology of the target physical device
            method(str, optional): used mapping method in ['mcts', 'sabre']
        """
        assert layout is not None, ValueError('No Layout provided for Mapping')
        assert method in ['mcts', 'sabre'], ValueError('Invalid mapping method')
        mapping_dict = {
            'mcts': MCTSMapping(layout=layout),
            'sabre': SABREMapping(layout=layout)
        }
        self.add_method(mapping_dict[method])

    def compile(self, circuit: Circuit):
        """ Compile the circuit with the given process

        Args:
            circuit(CompositeGate/Circuit): the target CompositeGate or Circuit

        Returns:
            CompositeGate/Circuit: the resulting CompositeGate or Circuit
        """
        logger.info("QCDA Now processing GateDecomposition.")
        circuit.gate_decomposition()
        for process in self.process:
            logger.info(f"QCDA Now processing {process.__class__.__name__}.")
            circuit = process.execute(circuit)

        return circuit

    def auto_compile(self, circuit: Circuit, quantum_machine_info: VirtualQuantumMachine):
        """ Auto-Compile the circuit with the given quantum machine info. Normally follow the steps:

        1. Optimization
        2. Mapping
        3. Gate Transfer
        4. Optimization

        Args:
            circuit (CompositeGate/Circuit): the target CompositeGate or Circuit
            quantum_machine_info (VirtualQuantumMachine): the information about target quantum machine.
        """
        qm_iset = quantum_machine_info.instruction_set
        qm_layout = quantum_machine_info.layout
        qm_process = []
        # Step 1: optimization algorithm for common circuit
        circuit.decomposition()
        circuit.flatten()
        if circuit.count_gate_by_gatetype(GateType.cx) == circuit.size():
            qm_process.append(CnotWithoutAncilla())
        else:
            gate_types = [gate.type for gate, _ in circuit.fast_gates]
            qm_process.append(self._choice_opt_algorithm(gate_types))

        # Step 2: Mapping if layout is not all-connected
        if qm_layout is not None:
            qm_process.append(SABREMapping(qm_layout))

        # Step 3: Gate Transfer by the given instruction set
        qm_process.append(GateTransform(qm_iset))

        # Step 4: Depending on the special instruction set gate, choice best optimization algorithm.
        iset_gtypes = qm_iset.gates
        qm_process.append(self._choice_opt_algorithm(iset_gtypes))

        # Step 5: Start the auto QCDA process:
        logger.info("QCDA Now processing GateDecomposition.")
        for process in qm_process:
            logger.info(f"QCDA Now processing {process.__class__.__name__}.")
            circuit = process.execute(circuit)

        return circuit

    def _choice_opt_algorithm(self, gate_types: list):
        clifford_only, extra_rz = True, False
        for gtype in gate_types:
            if gtype in (GateType.rz, GateType.ccx, GateType.ccz, GateType.t, GateType.tdg):
                extra_rz = True
            elif gtype not in CLIFFORD_GATE_SET:
                clifford_only = False
                break

        if clifford_only:
            return CliffordRzOptimization() if extra_rz else SymbolicCliffordOptimization()
        else:
            return CommutativeOptimization()
