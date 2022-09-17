import logging
import numpy as np

from QuICT.core.gate import *
from QuICT.simulation.state_vector import CircuitSimulator

from .canonical_QAE import amplitude_estimate as canonical_run
from .max_likely_QAE import amplitude_estimate as max_likely_run
from .FQAE import amplitude_estimate as fast_run
from .canonical_QAE import construct_circuit as canonical_circuit
from .FQAE import construct_circuit as fast_circuit
from .utility import StatePreparationInfo, OracleInfo

class QAE:
    _ALLOWED_MODES = {"canonical", "fast", "max_likely"}
    _RUN_METHOD_OF_MODE = {
        "canonical": canonical_run,
        "fast": fast_run,
        "max_likely": max_likely_run,
    }
    _CIRCUIT_METHOD_OF_MODE = {"canonical": canonical_circuit, "fast": fast_circuit}

    def __init__(
        self,
        mode,
        eps=0.1,
        simulator=CircuitSimulator(),
    ) -> None:
        self.mode = mode
        self.eps = eps
        self.simulator = simulator

    def run(
        self,
        oracle: OracleInfo = None,
        state_preparation: StatePreparationInfo = None,
    ):
        return QAE._RUN_METHOD_OF_MODE[self.mode](oracle=oracle, state_preparation=state_preparation,simulator=self.simulator)

    def circuit(
        self,
        oracle: OracleInfo = None,
        state_preparation: StatePreparationInfo = None,
    ):
        return QAE._CIRCUIT_METHOD_OF_MODE[self.mode](oracle=oracle, state_preparation=state_preparation)
