import logging
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import CircuitSimulator
from QuICT.qcda.synthesis.mct import MCTOneAux

class OracleInfo:
    def __init__(
        self,
        n=None,
        n_ancilla=None,
        S_chi=None,
        is_good_state=None,
        custom_grover_operator=None,
    ) -> None:
        self.n = n
        self.n_ancilla = n_ancilla
        self.S_chi = S_chi
        self.is_good_state = is_good_state
        self.custom_grover_operator = custom_grover_operator


class StatePreparationInfo:
    def __init__(self, n=None, n_ancilla=None, A=None, A_dagger=None) -> None:
        if A is None or A_dagger is None:
            logging.info("StatePreparationInfo: using default state preparation.")
            self.A = StatePreparationInfo.default_A
            self.A_dagger = StatePreparationInfo.default_A_dagger
            self.n_ancilla = 0
            self.n = n
        else:
            self.A = A
            self.A_dagger = A_dagger
            self.n_ancilla = n_ancilla
            self.n = n
        self.S_0 = StatePreparationInfo.default_S_0

    def default_A(n, controlled=False):
        # control on 0
        cgate = CompositeGate()
        if controlled:
            for i in range(1, 1 + n):
                CH | cgate([0, i])
        else:
            for i in range(n):
                H | cgate(i)
        return cgate

    def default_A_dagger(n, controlled=False):
        return StatePreparationInfo.default_A(n, controlled)

    def default_S_0(n, controlled=False):
        # control on 0
        cgate = CompositeGate()
        if controlled:
            indices = list(range(1,1+n))
            for i in indices:
                CX | cgate([0,i])
            CH | cgate([0,indices[n-1]])
            MCTOneAux().execute(n+2) | cgate
            CH | cgate([0,indices[n-1]])
            for i in indices:
                CX | cgate([0,i])
        else:
            for i in range(n):
                X | cgate(i)
            H | cgate(n-1)
            MCTOneAux().execute(n+1) | cgate
            H | cgate(n-1)
            for i in range(n):
                X | cgate(i)
        return cgate

