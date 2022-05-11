#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/4/27 下午5:50
# @Author  : Kaiqi Li
# @File    : proxy_simulator
import random
from collections import defaultdict
import numpy as np
from QuICT.simulation.cpu_simulator.cpu import CircuitSimulator
from QuICT.simulation.gpu_simulator.statevector_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
import cupy as cp

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.gpu_simulator import BasicGPUSimulator
from QuICT.utility import Proxy
from QuICT.ops.utils import LinAlgLoader
from QuICT.simulation.utils import GateGroup, GATE_TYPE_to_ID, MATRIX_INDEXES
from QuICT.simulation.gpu_simulator.multigpu_simulator.data_switch import DataSwitcher
from QuICT.qcda.synthesis import GateDecomposition


class MultiStateVectorSimulator:
    """
    The simulator which using multi-GPUs.

    Args:
        proxy (Proxy): The NCCL communicators.
        precision (str): The precision for the state vector, single precision means complex64,
            double precision means complex128.
        gpu_device_id (int): The GPU device ID.
        sync (bool): Sync mode or Async mode.
    """
    @property
    def vector(self):
        return self.simulator.vector

    def __init__(
        self,
        proxy: Proxy,
        device: str,
        **options
    ):
        self.proxy = DataSwitcher(proxy)
        self.simulator = CircuitSimulator(options) if device == "CPU" else \
            ConstantStateVectorSimulator(options)

    def run(
        self,
        circuit: Circuit
    ) -> np.ndarray:
        """ start simulator with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Returns:
            [array]: The state vector.
        """
        qubits = circuit.width()
        # initial state vector in simulator
        self.simulator.initial_state_vector(qubits)

        with cp.cuda.Device(self._device_id):
            for gate in self._gates:
                self.apply_gate(gate)

        return self.simulator.vector

    def _measure_operation(self, index):
        """ The algorithm for the Measure gate.

        Args:
            index (int): The target qubit of the applied quantum gate.

        Returns:
            [bool]: state 0 or state 1
        """
        default_parameters = (self._vector, self.total_qubits, self._sync)

        # Calculate the device's probability.
        prob_result = self._algorithm.Device_Prob_Calculator(
            index,
            self._vector,
            self.qubits,
            self.proxy.dev_id
        )

        # Combined the probability for all device
        total_prob = self._data_switcher.add_prob(prob_result)
        _0 = random.random()        # random mistake
        _1 = _0 > total_prob        # result in state 0 or state 1
        prob = total_prob.get()

        if index >= self.qubits:    # target index exceed the limit
            if _1:  # result in state 1
                alpha = np.float32(1 / np.sqrt(1 - prob)) if self._precision == np.complex64 else \
                    np.float64(1 / np.sqrt(1 - prob))

                if self.proxy.dev_id & (1 << (index - self.qubits)):
                    self._algorithm.Float_Multiply(
                        alpha,
                        *default_parameters
                    )
                else:
                    self._vector = cp.zeros_like(self._vector)
            else:       # result in state 0
                alpha = np.float32(1 / np.sqrt(prob)) if self._precision == np.complex64 else \
                    np.float64(1 / np.sqrt(prob))

                if self.proxy.dev_id & (1 << (index - self.qubits)):
                    self._vector = cp.zeros_like(self._vector)
                else:
                    self._algorithm.Float_Multiply(
                        alpha,
                        *default_parameters
                    )
        else:
            self._algorithm.MeasureGate_Apply(
                index,
                *default_parameters,
                multigpu_prob=prob
            )

        return _1

    def _reset_operation(self, index):
        """ The algorithm for the Reset gate.

        Args:
            index (int): The target qubit of the applied quantum gate.
        """
        default_parameters = (self._vector, self.total_qubits, self._sync)

        # Calculate the device's probability
        prob_result = self._algorithm.Device_Prob_Calculator(
            index,
            self._vector,
            self.qubits,
            self.proxy.dev_id
        )

        total_prob = self._data_switcher.add_prob(prob_result)
        prob = total_prob.get()

        if index >= self.qubits:
            alpha = np.float64(np.sqrt(prob))

            if alpha < 1e-6:
                destination = self.proxy.dev_id ^ (1 << (index - self.qubits))
                if not (self.proxy.dev_id & (1 << (index - self.qubits))):
                    self._vector = cp.zeros_like(self._vector)
                    self._data_switcher.all_switch(self._vector, destination)
            else:
                if self.proxy.dev_id & (1 << (index - self.qubits)):
                    self._vector = cp.zeros_like(self._vector)
                else:
                    alpha = np.float64(1 / alpha)
                    self._algorithm.Float_Multiply(
                        alpha,
                        *default_parameters
                    )
        else:
            self._algorithm.ResetGate_Apply(
                index,
                *default_parameters,
                multigpu_prob=prob
            )
