from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
import numpy as np


class QPE:
    def __init__(
        self,
        workbits: list,
        trickbits: list,
        workbits_state_preparation: CompositeGate,
        controlled_unitary,
        simulator=StateVectorSimulator()
    ) -> None:
        self.simulator = simulator
        self.trickbits = trickbits
        self.n = len(workbits)
        self.n_phase = len(trickbits)
        # contruct circuit
        circ = Circuit(self.n + self.n_phase)
        # Hadamard Transform
        for idx in trickbits:
            H | circ(idx)
        # controlled unitary, EXPECT idx 0 to be the control bit
        workbits_state_preparation | circ(workbits)
        for k in range(self.n_phase):
            controlled_unitary(1<<(self.n_phase-1-k)) | circ(
                [trickbits[k]] + workbits
            )
        # IQFT
        for k in range(self.n_phase // 2):
            Swap | circ([trickbits[k], trickbits[self.n_phase - 1 - k]])
        IQFT.build_gate(self.n_phase) | circ(trickbits)
        # for idx in trickbits:
        #     Measure | circ(idx)
        self._circuit_cache = circ

    def circuit(self):
        return self._circuit_cache
    
    def run(self, shots = 100):
        self.simulator.run(self.circuit())
        result = self.simulator.sample(shots)
        observed = bin(np.argmax(np.array(result)))[2:].rjust(self.n+self.n_phase, '0')
        trickbits_observed = ""
        for bit in self.trickbits:
            trickbits_observed += observed[bit]
        print(trickbits_observed)
        return int(trickbits_observed, 2) / (1<<self.n_phase)

