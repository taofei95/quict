"""
Decompose gates except for BasicGates in a CompositeGate or a Circuit
"""

import numpy as np

from QuICT.core import BasicGate, Circuit, ComplexGate, CompositeGate, UnitaryGate
from QuICT.qcda.synthesis.unitary_transform import UnitaryTransform
from QuICT.tools.interface import OPENQASMInterface
from .._synthesis import Synthesis

class GateDecomposition(Synthesis):
    """ Gate decomposition method
    """
    @classmethod
    def execute(cls, objective):
        """
        Decompose gates except for BasicGates in a CompositeGate or a Circuit
        to BasicGates with the `build_gate` method if it is implemented.
        Otherwise the `UnitaryTransform` would be used. 

        Args:
            objective: objective of GateDecomposition, the following types are supported.
                1. str: the objective is the path of an OPENQASM file
                2. numpy.ndarray: the objective is a unitary matrix
                3. Circuit: the objective is a Circuit
                4. CompositeGate: the objective is a CompositeGate

        Returns:
            CompositeGate: Decomposed CompositeGate

        Raises:
            If the objective could not be resolved as any of the above types.
        """
        # Load the objective as raw_gates
        if isinstance(objective, np.ndarray):
            raw_gates, _ = UnitaryTransform.execute(objective)
            # No ComplexGates needed to be checked in this case
            return raw_gates

        if isinstance(objective, str):
            qasm = OPENQASMInterface.load_file(objective)
            if qasm.valid_circuit:
                # FIXME: no circuit here
                circuit = qasm.circuit
                raw_gates = CompositeGate(circuit)
            else:
                raise ValueError("Invalid qasm file!")

        if isinstance(objective, Circuit):
            raw_gates = CompositeGate(objective)

        if isinstance(objective, CompositeGate):
            raw_gates = CompositeGate(objective)
        
        assert isinstance(raw_gates, CompositeGate), TypeError('Invalid objective!')

        # Decomposition of complex gates
        gates = CompositeGate()
        for gate in raw_gates:
            if isinstance(gate, UnitaryGate):
                gate_decomposed, _ = UnitaryTransform.execute(gate.compute_matrix, mapping=gate.targs)
                gates.extend(gate_decomposed)
            # Be aware of the order here, since ComplexGate is inherited from BasicGate
            elif isinstance(gate, ComplexGate):
                gates.extend(gate.build_gate())
            elif isinstance(gate, BasicGate):
                gates.append(gate)
            else:
                raise ValueError('Unknown gate encountered')

        return gates
