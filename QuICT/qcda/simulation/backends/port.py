from ctypes import cdll
import os
from QuICT.core import *

import os
import importlib.util
from typing import List, Union
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
    def __init__(self, gate_name: str, affect_args: List[int], data_ptr: List[complex]):
        pass


class CircuitSimulator:
    def __init__(self, qubit_num: int):
        pass

    def name(self) -> str:
        pass

    def run(self, gate_desc_vec: List[GateDescription]) -> np.ndarray:
        pass


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


def gate_to_desc(gate: BasicGate) -> GateDescription:
    gate_type = gate.type()
    if gate_type in special_x:
        return sim_back_bind.GateDescription(
            "special_x",
            list(gate.affectArgs),
            list([])
        )
    elif gate_type in special_h:
        return sim_back_bind.GateDescription(
            "special_h",
            list(gate.affectArgs),
            list([])
        )
    elif gate_type in diag_1:
        return sim_back_bind.GateDescription(
            "diag_1",
            list(gate.affectArgs),
            list(np.diag(gate.compute_matrix))
        )
    elif gate_type in diag_2:
        return sim_back_bind.GateDescription(
            "diag_2",
            list(gate.affectArgs),
            list(np.diag(gate.compute_matrix))
        )
    elif gate_type in unitary_1:
        return sim_back_bind.GateDescription(
            "unitary_1",
            list(gate.affectArgs),
            list(gate.compute_matrix.flatten())
        )
    elif gate_type in unitary_2:
        return sim_back_bind.GateDescription(
            "unitary_2",
            list(gate.affectArgs),
            list(gate.compute_matrix.flatten())
        )
    elif gate_type in ctrl_diag:
        return sim_back_bind.GateDescription(
            "ctrl_diag",
            list(gate.affectArgs),
            list(np.diag(gate.compute_matrix)[2:].copy())
        )
    elif gate_type in ctrl_unitary:
        return sim_back_bind.GateDescription(
            "ctrl_unitary",
            list(gate.affectArgs),
            list(gate.compute_matrix[2:, 2:].copy().flatten())
        )
    else:
        NotImplementedError(f"No implementation for {gate.name}")


def run_simulation(circuit: Union[Circuit, CompositeGate]) -> np.ndarray:
    circuit_simulator: CircuitSimulator = sim_back_bind.CircuitSimulator(circuit.circuit_width())
    gate_desc_vec: List[GateDescription] = []
    for gate in circuit.gates:
        gate_desc_vec.append(gate_to_desc(gate))
    return circuit_simulator.run(gate_desc_vec)
