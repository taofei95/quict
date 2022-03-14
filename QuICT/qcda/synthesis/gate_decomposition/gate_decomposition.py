"""
Decompose gates except for BasicGates in a CompositeGate or a Circuit
"""

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate, CompositeGate, UnitaryGate
from QuICT.qcda.synthesis.unitary_transform import UnitaryTransform
from QuICT.tools.interface import OPENQASMInterface
from .._synthesis import Synthesis


class GateDecomposition(Synthesis):
    """ Gate decomposition method """
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
            return raw_gates

        if isinstance(objective, str):
            qasm = OPENQASMInterface.load_file(objective)
            if qasm.valid_circuit:
                # FIXME: no circuit here
                circuit = qasm.circuit
                raw_gates = CompositeGate(gates=circuit.gates)
            else:
                raise ValueError("Invalid qasm file!")

        if isinstance(objective, Circuit):
            raw_gates = CompositeGate(gates=objective.gates)

        if isinstance(objective, CompositeGate):
            raw_gates = CompositeGate(gates=objective.gates)

        assert isinstance(raw_gates, CompositeGate), TypeError('Invalid objective!')

        # Decomposition of complex gates
        gates = CompositeGate()
        for gate in raw_gates:
            if isinstance(gate, UnitaryGate):
                gate_decomposed, _ = UnitaryTransform.execute(gate.matrix, mapping=gate.targs)
                gates.extend(gate_decomposed)
            elif isinstance(gate, BasicGate):
                try:
                    gates.extend(gate.build_gate())
                except:
                    gates.append(gate)
            else:
                raise ValueError('Unknown gate encountered')

        return gates
