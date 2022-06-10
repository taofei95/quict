#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/4/27 下午5:50
# @Author  : Kaiqi Li
# @File    : proxy_simulator
import random
import cupy as cp
import numpy as np

from QuICT.simulation.state_vector import CircuitSimulator, ConstantStateVectorSimulator

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate, GateType
from QuICT.core.operator import *
from QuICT.utility import Proxy
from QuICT.simulation.state_vector.gpu_simulator.multigpu_simulator.data_switch import DataSwitcher
from QuICT.ops.gate_kernel.gate_function import Device_Prob_Calculator


class MultiNodesSimulator:
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
        device: str = "GPU",
        gpu_id: int = 0,
        **options
    ):
        self.proxy = proxy
        self._data_switcher = DataSwitcher(self.proxy)
        self.simulator = CircuitSimulator(options) if device == "CPU" else \
            ConstantStateVectorSimulator(gpu_device_id=gpu_id)

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
        self.simulator.initial_state_vector(qubits, (self._data_switcher.id != 0))
        self._pipeline = circuit.gates
        while len(self._pipeline) > 0:
            op = self._pipeline.pop(0)

            if isinstance(op, BasicGate):
                self.simulator.apply_gate(op)
            elif isinstance(op, DeviceTrigger):
                related_gate = op.mapping(self._data_switcher.id)
                self._pipeline = related_gate.gates + self._pipeline
            elif isinstance(op, DataSwitch):
                self._data_switcher(op, self.vector)
            elif isinstance(op, Multiply):
                self.simulator.apply_multiply(op.value)
            elif isinstance(op, SpecialGate):
                self.apply_specialgate(op)
            else:
                raise TypeError("Unsupportted operator in Multi-Nodes Simulator.")

        return self.simulator.vector

    def apply_specialgate(self, op: SpecialGate):
        
        if self._data_switcher.id & op.proxy_idx:
            prob = 0
        else:
            # prob calculation for all switch
            prob = Device_Prob_Calculator(self.vector)

        print(prob)
        print(type(prob))
        total_prob = self._data_switcher.add_prob(prob)
        total_prob = total_prob.get()

        if op.proxy_idx != -1:
            if op.type == GateType.measure:
                self._apply_measure(op.proxy_idx, total_prob)
            elif op.type == GateType.reset:
                self._apply_reset(op.proxy_idx, total_prob)
        else:
            self.simulator.apply_specialgate(op.targ, op.type, total_prob)

    def _apply_measure(self, proxy_idx: int, prob: float):
        """ The algorithm for the Measure gate.

        Args:
            index (int): The target qubit of the applied quantum gate.

        Returns:
            [bool]: state 0 or state 1
        """
        _0 = random.random()        # random mistake
        if _0 > prob:  # result in state 1
            alpha = np.float32(1 / np.sqrt(1 - prob)) if self.simulator._precision == np.complex64 else \
                np.float64(1 / np.sqrt(1 - prob))

            if self._data_switcher.id & proxy_idx:
                self.simulator.apply_multiply(alpha)
            else:
                self.vector = cp.zeros_like(self.vector)
        else:       # result in state 0
            alpha = np.float32(1 / np.sqrt(prob)) if self.simulator._precision == np.complex64 else \
                np.float64(1 / np.sqrt(prob))

            if self._data_switcher.id & proxy_idx:
                self.vector = cp.zeros_like(self.vector)
            else:
                self.simulator.apply_multiply(alpha)

        return _0 > prob

    def _apply_reset(self, proxy_idx: int, prob: float):
        """ The algorithm for the Reset gate.

        Args:
            index (int): The target qubit of the applied quantum gate.
        """
        alpha = np.float64(np.sqrt(prob))
        if alpha < 1e-6:
            destination = self._data_switcher.id ^ proxy_idx
            if not (self._data_switcher.id & proxy_idx):
                self.vector = cp.zeros_like(self.vector)
                self._data_switcher.all_switch(self.vector, destination)
        else:
            if self._data_switcher.id & proxy_idx:
                self.vector = cp.zeros_like(self.vector)
            else:
                self.simulator.apply_multiply(np.float64(1 / alpha))
