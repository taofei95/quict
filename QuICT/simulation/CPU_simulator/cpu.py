import os
from QuICT.core import *

import warnings
import os
import importlib.util
from typing import List, Union, Iterable, Tuple
import numpy as np

cur_path = os.path.dirname(os.path.abspath(__file__))

mod_name = "sim_back_bind"
mod_path = "sim_back_bind"

for file in os.listdir(cur_path):
    # print(file)
    if file.startswith(mod_path):
        mod_path = os.path.join(cur_path, file)

spec = importlib.util.spec_from_file_location(mod_name, mod_path)
sim_back_bind = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_back_bind)


class GateDescription:
    """An interface used for type hints. This class is actually implemented in C++ side.
    """

    def __init__(self, gate_name: str, affect_args: Iterable[int], data_ptr: Iterable[complex]):
        pass


# sim_back_bind.GateDescription: GateDescription.__class__

special_x = (GATE_ID["X"],)
special_h = (GATE_ID["H"],)
diag_1 = (
    GATE_ID["S"],
    GATE_ID["S_dagger"],
    GATE_ID["Z"],
    GATE_ID["ID"],
    GATE_ID["U1"],
    GATE_ID["Rz"],
    GATE_ID["T"],
    GATE_ID["T_dagger"],
    GATE_ID["Phase"],  # TODO: optimize this with special_phase
)

unitary_1 = (
    GATE_ID["Y"],
    GATE_ID["SX"],
    GATE_ID["SY"],
    GATE_ID["SW"],
    GATE_ID["U2"],
    GATE_ID["U3"],
    GATE_ID["Rx"],
    GATE_ID["Ry"],
)

diag_2 = (
    GATE_ID["Rzz"],
)

unitary_2 = (
    GATE_ID["FSim"],
    GATE_ID["Rxx"],
    GATE_ID["Ryy"],
    GATE_ID["Swap"],  # Maybe this could be optimized
)

ctrl_diag = (
    GATE_ID["CZ"],
    GATE_ID["CRZ"],
    GATE_ID["CU1"],
)

ctrl_unitary = (
    GATE_ID["CX"],  # TODO: Optimize this with special_cx
    GATE_ID["CY"],
    GATE_ID["CH"],
    GATE_ID["CU3"],
)

measure_gate = tuple([GATE_ID["measure"]])


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
    gate_type = gate.type()
    if gate_type in special_x:
        return [sim_back_bind.GateDescription(
            "special_x",
            list(gate.affectArgs),
            list([])
        )]
    elif gate_type in special_h:
        return [sim_back_bind.GateDescription(
            "special_h",
            list(gate.affectArgs),
            list([])
        )]
    elif gate_type in diag_1:
        return [sim_back_bind.GateDescription(
            "diag_1",
            list(gate.affectArgs),
            list(np.diag(gate.compute_matrix))
        )]
    elif gate_type in diag_2:
        return [sim_back_bind.GateDescription(
            "diag_2",
            list(gate.affectArgs),
            list(np.diag(gate.compute_matrix))
        )]
    elif gate_type in unitary_1:
        return [sim_back_bind.GateDescription(
            "unitary_1",
            list(gate.affectArgs),
            list(gate.compute_matrix.flatten())
        )]
    elif gate_type in unitary_2:
        return [sim_back_bind.GateDescription(
            "unitary_2",
            list(gate.affectArgs),
            list(gate.compute_matrix.flatten())
        )]
    elif gate_type in ctrl_diag:
        return [sim_back_bind.GateDescription(
            "ctrl_diag",
            list(gate.affectArgs),
            list(np.diag(gate.compute_matrix)[2:].copy())
        )]
    elif gate_type in ctrl_unitary:
        return [sim_back_bind.GateDescription(
            "ctrl_unitary",
            list(gate.affectArgs),
            list(gate.compute_matrix[2:, 2:].copy().flatten())
        )]
    elif gate_type in measure_gate:
        return [sim_back_bind.GateDescription(
            "measure",
            list(gate.affectArgs),
            list([])
        )]
    elif isinstance(gate, ComplexGate):
        # Try build gate to simpler gates
        result = []
        for simple_gate in gate.build_gate():
            result.extend(gate_to_desc(simple_gate))
        return result
    else:
        NotImplementedError(f"No implementation for {gate.name}")


class CircuitSimulator:
    """An interface used for type hints. This class is actually implemented in C++ side.
    """

    def __init__(self):
        self._instance = sim_back_bind.CircuitSimulator()

    def name(self) -> str:
        """

        Returns
        -------
        The name of circuit simulator.
        """
        return self._instance.name()

    def _run(self, circuit: Circuit, keep_state: bool = False) -> Tuple[np.ndarray, List[int]]:
        """Run simulation by gate description sequence and return measure gate results.

        Parameters
        ----------
        circuit:
            quantum circuit to be simulated
        keep_state:
            start simulation on previous result
        """
        warnings.warn(
            message="Attention! You are using a working-in-process version of circuit simulator!",
            category=Warning,
            stacklevel=1)
        gate_desc_vec: List[GateDescription] = []
        for gate in circuit.gates:
            gate_desc_vec.extend(gate_to_desc(gate))
        return self._instance.run(circuit.circuit_width(), gate_desc_vec, keep_state)

    def run(self, circuit: Circuit, keep_state: bool = False) -> np.ndarray:
        """Run simulation by gate description sequence and return measure gate results.

        Parameters
        ----------
        circuit:
            quantum circuit to be simulated
        keep_state:
            start simulation on previous result
        """
        amplitude, _ = self._run(circuit, keep_state)
        return amplitude

    def sample(self, circuit: Circuit) -> List[int]:
        """Appending measure gates to end of circuit if not presented then
        apply measurement. Before calling this method, one should call `run`
        method first.

        Parameters
        ----------
        circuit:
            quantum circuit to be simulated
        """
        return self._instance.sample(circuit.circuit_width())
