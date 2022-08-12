import warnings
import os
import importlib.util
from typing import List, Union, Iterable, Tuple
import numpy as np
from copy import deepcopy

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate, Measure, GateType
from QuICT.core.operator import Trigger


cur_path = os.path.dirname(os.path.abspath(__file__))

mod_name = "sim_back_bind"
mod_path = "sim_back_bind"

for file in os.listdir(cur_path):
    if file.startswith(mod_path):
        mod_path = os.path.join(cur_path, file)

spec = importlib.util.spec_from_file_location(mod_name, mod_path)
sim_back_bind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_back_bind)


def get_gate_qubit_pos(gate: BasicGate) -> List[int]:
    args = deepcopy(gate.cargs)
    args.extend(deepcopy(gate.targs))
    return args


class GateDescription:
    """An interface used for type hints. This class is actually implemented in C++ side.
    """

    def __init__(
        self, gate_name: str, affect_args: Iterable[int], data_ptr: Iterable[complex]
    ):
        pass


# sim_back_bind.GateDescription: GateDescription.__class__

special_x = (GateType.x,)
special_h = (GateType.h,)
diag_1 = (
    GateType.s,
    GateType.sdg,
    GateType.z,
    GateType.id,
    GateType.u1,
    GateType.rz,
    GateType.t,
    GateType.tdg,
    GateType.phase,  # TODO: optimize this with special_phase
)

# The unitary_x category is some gates treated as unitary.
# The real unitary gates are not included.
unitary_1 = (
    GateType.y,
    GateType.sx,
    GateType.sy,
    GateType.sw,
    GateType.u2,
    GateType.u3,
    GateType.rx,
    GateType.ry,
)

diag_2 = (GateType.rzz,)

unitary_2 = (
    GateType.fsim,
    GateType.rxx,
    GateType.ryy,
    GateType.swap,  # Maybe this could be optimized
)

ctrl_diag = (
    GateType.cz,
    GateType.crz,
    GateType.cu1,
)

ctrl_unitary = (
    GateType.cx,  # TODO: Optimize this with special_cx
    GateType.cy,
    GateType.ch,
    GateType.cu3,
)

measure_gate = (GateType.measure,)


def gate_to_desc(gate: BasicGate) -> List[GateDescription]:
    """Helper function to create GateDescription from a quantum gate.

    Parameters
    ----------
    gate:
        A quantum gate.

    Returns
    -------
    Simple GateDescription for input gate.
    """
    gate_type = gate.type
    if gate_type in special_x:
        return [
            sim_back_bind.GateDescription(
                "special_x", list(get_gate_qubit_pos(gate)), list([])
            )
        ]
    elif gate_type in special_h:
        return [
            sim_back_bind.GateDescription(
                "special_h", list(get_gate_qubit_pos(gate)), list([])
            )
        ]
    elif gate_type in diag_1:
        return [
            sim_back_bind.GateDescription(
                "diag_1", list(get_gate_qubit_pos(gate)), list(np.diag(gate.matrix))
            )
        ]
    elif gate_type in diag_2:
        return [
            sim_back_bind.GateDescription(
                "diag_2", list(get_gate_qubit_pos(gate)), list(np.diag(gate.matrix))
            )
        ]
    elif gate_type in unitary_1:
        return [
            sim_back_bind.GateDescription(
                "unitary_1", list(get_gate_qubit_pos(gate)), list(gate.matrix.flatten())
            )
        ]
    elif gate_type in unitary_2:
        return [
            sim_back_bind.GateDescription(
                "unitary_2", list(get_gate_qubit_pos(gate)), list(gate.matrix.flatten())
            )
        ]
    elif gate_type in ctrl_diag:
        return [
            sim_back_bind.GateDescription(
                "ctrl_diag",
                list(get_gate_qubit_pos(gate)),
                list(np.diag(gate.matrix)[2:].copy()),
            )
        ]
    elif gate_type in ctrl_unitary:
        return [
            sim_back_bind.GateDescription(
                "ctrl_unitary",
                list(get_gate_qubit_pos(gate)),
                list(gate.matrix[2:, 2:].copy().flatten()),
            )
        ]
    elif gate_type in measure_gate:
        return [
            sim_back_bind.GateDescription(
                "measure", list(get_gate_qubit_pos(gate)), list([])
            )
        ]
    elif gate_type == GateType.unitary and gate.targets <= 2:
        return [
            sim_back_bind.GateDescription(
                f"unitary_{gate.targets}",
                list(get_gate_qubit_pos(gate)),
                list(gate.matrix.flatten()),
            )
        ]
    elif hasattr(gate, "build_gate"):
        # Try build gate to simpler gates
        result = []
        cgate = gate.build_gate()
        for simple_gate in cgate.gates:
            result.extend(gate_to_desc(simple_gate))
        return result
    else:
        NotImplementedError(f"No implementation for {gate.name}")


class CircuitSimulator:
    """An interface used for type hints. This class is actually implemented in C++ side.
    """

    def __init__(self):
        self._circuit = None
        self._instance = sim_back_bind.CircuitSimulator()
        self._gate_desc_vec: List[GateDescription] = []
        self._pending_gates: List[BasicGate] = []

    def name(self) -> str:
        """

        Returns
        -------
        The name of circuit simulator.
        """
        return self._instance.name()

    @classmethod
    def _map_measure(cls, circuit: Circuit, measure_raw: List[int]) -> List[List[int]]:
        mid_map = []
        for gate in circuit.gates:
            if isinstance(gate, BasicGate) and gate.type == GateType.measure:
                mid_map.append(gate.targ)

        measure: List[List[int]] = [[] for _ in range(circuit.width())]
        for idx, elem in enumerate(mid_map):
            measure[elem].append(measure_raw[idx])
            circuit.qubits[elem].measured = measure_raw[idx]

        return measure

    def apply_gate(self, gate: BasicGate):
        self._gate_desc_vec.append(deepcopy(gate))

    def _run(
        self, circuit: Union[Circuit, None], keep_state: bool = False
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """Run simulation by gate description sequence and return measure gate results.

        Parameters
        ----------
        circuit:
            Quantum circuit to be simulated. If `None` is passed, then all previous gates added by
            `apply_gate` would be executed.
        keep_state:
            Start simulation on previous result
        """
        warnings.warn(
            message="Attention! You are using a working-in-process version of circuit simulator!",
            category=Warning,
            stacklevel=1,
        )
        if circuit:
            qubits = circuit.width()
            gate_set = []
            for gate in circuit.gates:
                if isinstance(gate, BasicGate):
                    gate_set.append(deepcopy(gate))
                else:
                    gate_set.append(gate)

            gate_desc_vec: List[GateDescription] = []
            idx = 0
            while gate_set:
                gate = gate_set.pop(0)
                if isinstance(gate, Trigger):
                    for targ in gate.targs:
                        mgate = Measure & targ
                        gate_desc_vec.extend(gate_to_desc(mgate))

                    _, measure_raw = self._instance.run(
                        qubits, gate_desc_vec, keep_state
                    )
                    gate_desc_vec = []
                    measured_state, keep_state = 0, True
                    for mstate in measure_raw:
                        measured_state << 1
                        measured_state += mstate

                    cgate = gate.mapping(measured_state)
                    if cgate is not None:
                        cp = cgate.checkpoint
                        position = 0 if cp is None else circuit.find_position(cp) - idx
                        gate_set = (
                            gate_set[:position]
                            + deepcopy(cgate.gates)
                            + gate_set[position:]
                        )
                else:
                    gate_desc_vec.extend(gate_to_desc(gate))

                idx += 1

        self._gate_desc_vec = gate_desc_vec

        amplitude, measure_raw = self._instance.run(
            circuit.width(), self._gate_desc_vec, keep_state
        )
        measure = self._map_measure(circuit, measure_raw)
        self._gate_desc_vec.clear()
        return amplitude, measure

    def run(
        self,
        circuit: Union[Circuit, None],
        keep_state: bool = False,
        output_measure_res: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[List[int]]]]:
        """Run simulation by gate description sequence and return measure gate results.

        Parameters
        ----------
        circuit:
            Quantum circuit to be simulated. If `None` is passed, then all previous gates added by
            `apply_gate` would be executed.
        keep_state:
            Start simulation on previous result
        """
        amplitude, measure = self._run(circuit, keep_state)
        self._circuit = circuit

        if output_measure_res:
            return amplitude, measure
        else:
            return amplitude

    def sample(self, shots: int = 1) -> List[int]:
        assert self._circuit is not None

        measure_accumulated = [0 for _ in range(self._circuit.width())]

        for _ in range(shots):
            # C++ simulator will automatically copy a state vector for sampling.
            # Sample operation never affects the state vector itself.
            measure_raw = self._instance.sample(self._circuit.width())

            for idx, measured in enumerate(measure_raw):
                measure_accumulated[idx] += measured

        return measure_accumulated