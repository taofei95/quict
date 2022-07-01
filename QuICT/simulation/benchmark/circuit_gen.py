import numpy as np

from QuICT.core import *
from QuICT.core.gate import *
from random import choice, choices, sample, uniform
from typing import List, Iterable, Tuple, Callable

DEFAULT_SCALE_SMALL = [
    1,
    2,
    3,
    4,
    5,
]
DEFAULT_SCALE_MEDIUM = [
    13,
    14,
    15,
    16,
    17,
    18,
    19,
]
DEFAULT_SCALE_LARGE = [
    30,
    35,
    40,
    45,
    50,
]


def parse_scales(scale: str) -> List[int]:
    if scale == "small":
        return DEFAULT_SCALE_SMALL
    elif scale == "medium":
        return DEFAULT_SCALE_MEDIUM
    elif scale == "large":
        return DEFAULT_SCALE_LARGE
    else:
        raise ValueError("Only small/medium/large are allowed to choose circuit scale.")


def default_size(scale: int) -> int:
    return 40 * scale


SINGLE_BIT_GATE = (
    # No param
    GateType.x,
    GateType.y,
    GateType.z,
    GateType.h,
    GateType.s,
    GateType.sdg,
    # GateType.sx,
    # GateType.sy,
    # GateType.sw,
    GateType.t,
    GateType.tdg,
    # W/ param
    GateType.u1,
    GateType.u2,
    GateType.u3,
    GateType.rx,
    GateType.ry,
    GateType.rz,
    # GateType.phase,
)

DIAG_1_GATE = (
    GateType.s,
    GateType.sdg,
    GateType.z,
    GateType.id,
    GateType.u1,
    GateType.rz,
    GateType.t,
    GateType.tdg,
    # GateType.phase,
)

# DIAG_2_GATE = (GateType.Rzz,)
DIAG_2_GATE = tuple()

UNITARY_1_GATE = (
    GateType.y,
    # GateType.sx,
    # GateType.sy,
    # GateType.sw,
    GateType.u2,
    GateType.u3,
    GateType.rx,
    GateType.ry,
)

UNITARY_2_GATE = (
    # GateType.fsim,
    # GateType.Rxx,
    # GateType.Ryy,
    GateType.swap,
)

CTRL_DIAG_GATE = (
    GateType.cz,
    GateType.crz,
    GateType.cu1,
)

CTRL_UNITARY_GATE = (
    GateType.cx,
    GateType.cy,
    GateType.ch,
    GateType.cu3,
)


# DIAG_GATE = list(DIAG_1_GATE)
# DIAG_GATE.extend(DIAG_2_GATE)
# DIAG_GATE = tuple(DIAG_GATE)


def populate_random_param(gate: BasicGate) -> BasicGate:
    g = gate
    if gate.params > 0:
        params = []
        for _ in range(gate.params):
            params.append(uniform(0, 2 * np.pi))
        g = gate(*params)
    return g


def random_single_bit_gate() -> BasicGate:
    t = choice(SINGLE_BIT_GATE)
    gate = build_gate(t, [0])
    return populate_random_param(gate)


def random_diag_gate() -> BasicGate:
    t = choice(range(len(DIAG_1_GATE) + len(DIAG_2_GATE)))
    if t < len(DIAG_1_GATE):
        gate = build_gate(DIAG_1_GATE[t], [0])
    else:
        gate = build_gate(DIAG_2_GATE[t - len(DIAG_1_GATE)], [0, 1])
    return populate_random_param(gate)


def random_ctrl_diag_gate() -> BasicGate:
    t = choice(CTRL_DIAG_GATE)
    gate = build_gate(t, [0, 1])
    return populate_random_param(gate)


def random_unitary_gate() -> BasicGate:
    t = choice(range(len(UNITARY_1_GATE) + len(UNITARY_2_GATE)))
    if t < len(UNITARY_1_GATE):
        gate = build_gate(UNITARY_1_GATE[t], [0])
    else:
        gate = build_gate(UNITARY_2_GATE[t - len(UNITARY_1_GATE)], [0, 1])
    return populate_random_param(gate)


def random_ctrl_unitary_gate() -> BasicGate:
    t = choice(CTRL_UNITARY_GATE)
    gate = build_gate(t, [0, 1])
    return populate_random_param(gate)


class CircuitFactory:
    @staticmethod
    def default(
        scale: str, gate_generator: Callable[[], BasicGate]
    ) -> Iterable[Circuit]:
        qubit_scales = parse_scales(scale)
        for qubit_num in qubit_scales:
            circ = Circuit(qubit_num)
            for _ in range(default_size(qubit_num)):
                g = gate_generator()
                if g.controls + g.targets > qubit_num:
                    continue
                g | circ(sample(range(qubit_num), k=g.controls + g.targets))
            yield circ

    @staticmethod
    def qft(scale: str) -> Iterable[Circuit]:
        qubit_scales = parse_scales(scale)
        for qubit_num in qubit_scales:
            circ = Circuit(qubit_num)
            QFT.build_gate(qubit_num) | circ
            yield circ
