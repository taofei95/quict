import cupy as cp
import numpy as np
from typing import Union

from QuICT.core import Circuit
from QuICT.core.operator import Trigger
from QuICT.core.gate import BasicGate, CompositeGate
from QuICT.core.utils import GateType
from QuICT.simulation.utils import GateSimulator


class Differentiator:
    __DEVICE = ["CPU", "GPU"]
    __PRECISION = ["single", "double"]

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, circuit):
        self._circuit = circuit

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, vec):
        self._vector = self._gate_calculator.validate_state_vector(vec, self._qubits)

    @property
    def device(self):
        return self._device_id

    def __init__(
        self,
        device: str = "GPU",
        precision: str = "double",
        gpu_device_id: int = 0,
        sync: bool = True,
    ):
        if device not in self.__DEVICE:
            raise ValueError("Differentiator.device", "[CPU, GPU]", device)

        if precision not in self.__PRECISION:
            raise ValueError("Differentiator.precision", "[single, double]", precision)

        self._device = device
        self._precision = precision
        self._device_id = gpu_device_id
        self._sync = sync
        self._gate_calculator = GateSimulator(
            self._device, self._precision, self._device_id, self._sync
        )

    def __call__(
        self, circuit: Circuit, state_vector: np.ndarray, params, expectation_op
    ):
        raise NotImplementedError

    def initial_state_vector(self, all_zeros: bool = False):
        """ Initial qubits' vector states. """
        if not all_zeros:
            self._vector = self._gate_calculator.get_allzero_state_vector(self._qubits)
        else:
            self._vector = self._gate_calculator.get_empty_state_vector(self._qubits)

    def _apply_gate(self, gate: BasicGate, qidxes: list):
        """ Depending on the given quantum gate, apply the target algorithm to calculate the state vector.

        Args:
            gate (Gate): the quantum gate in the circuit.
        """
        gate_type = gate.type
        if gate_type in [GateType.measure, GateType.reset]:
            raise NotImplementedError
        else:
            self._gate_calculator.apply_gate(gate, qidxes, self._vector, self._qubits)

    def _apply_compositegate(self, gate: CompositeGate, qidxes: list):
        """ Depending on the given quantum gate, apply the target algorithm to calculate the state vector.

        Args:
            gate (Gate): the quantum gate in the circuit.
        """
        qidxes_mapping = {}
        cgate_qlist = gate.qubits
        for idx, cq in enumerate(cgate_qlist):
            qidxes_mapping[cq] = qidxes[idx]

        for cgate, cg_idx, size in gate.fast_gates:
            real_qidx = [qidxes_mapping[idx] for idx in cg_idx]
            if size > 1:
                self._apply_compositegate(cgate, real_qidx)
            else:
                self._apply_gate(cgate, real_qidx)

    # TODO: refactoring later, multi-gpu kernel function
    def apply_multiply(self, value: Union[float, np.complex]):
        """ Deal with Operator <Multiply>

        Args:
            value (Union[float, complex]): The multiply value apply to state vector.
        """
        from QuICT.ops.gate_kernel import float_multiply, complex_multiply

        default_parameters = (self._vector, self._qubits, self._sync)
        if isinstance(value, float):
            float_multiply(value, *default_parameters)
        else:
            if self._precision == np.complex64:
                value = np.complex64(value)

            complex_multiply(value, *default_parameters)

    def apply_zeros(self):
        """ Set state vector to be zero. """
        self._vector = self._gate_calculator.get_empty_state_vector(self._qubits)
