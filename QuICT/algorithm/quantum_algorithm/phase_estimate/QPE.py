from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, H, Swap, IQFT
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.tools import Logger
import numpy as np


logger = Logger("QPE")


class QPE:
    def __init__(self, simulator=StateVectorSimulator()) -> None:
        self.simulator = simulator
        self._circuit_cache = None

    def circuit(
        self,
        workbits: list,
        trickbits: list,
        workbits_state_preparation: CompositeGate,
        controlled_unitary,
    ):
        """QPE circuit construction.

        Args:
            workbits (list): indices for working qubits
            trickbits (list): indices for phase estimation ancillae
            workbits_state_preparation (CompositeGate): composite gate that prepares initial state in working qubits
            controlled_unitary (int->CompositeGate): a function that returns a composite gate, input parameter as its\
            repetition times

        Returns:
            Circuit: the QPE circuit (without measure)
        """
        n = len(workbits)
        n_phase = len(trickbits)
        # contruct circuit
        circ = Circuit(n + n_phase)
        # Hadamard Transform
        for idx in trickbits:
            H | circ(idx)
        # controlled unitary, EXPECT idx 0 to be the control bit
        workbits_state_preparation | circ(workbits)
        for k in range(n_phase):
            controlled_unitary(1 << (n_phase - 1 - k)) | circ([trickbits[k]] + workbits)
        # IQFT
        for k in range(n_phase // 2):
            Swap | circ([trickbits[k], trickbits[n_phase - 1 - k]])
        IQFT.build_gate(n_phase) | circ(trickbits)
        return circ

    def run(
        self,
        workbits: list,
        trickbits: list,
        workbits_state_preparation: CompositeGate,
        controlled_unitary,
        shots=100,
    ):
        self.simulator.run(
            self.circuit(
                workbits, trickbits, workbits_state_preparation, controlled_unitary
            )
        )
        result = self.simulator.sample(shots)
        n = len(workbits)
        n_phase = len(trickbits)
        observed = bin(np.argmax(np.array(result)))[2:].rjust(n + n_phase, "0")
        trickbits_observed = ""
        for bit in trickbits:
            trickbits_observed += observed[bit]
        logger.info(f"most possible trickbits: {trickbits_observed}")
        return int(trickbits_observed, 2) / (1 << n_phase)
