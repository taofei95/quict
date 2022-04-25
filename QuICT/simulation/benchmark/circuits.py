from QuICT.core import *
from QuICT.core.gate import *
from typing import List, Iterable, Tuple, Callable

DEFAULT_SCALE_SMALL = (1, 2, 3, 4, 5,)
DEFAULT_SCALE_MEDIUM = (15, 16, 17,)
DEFAULT_SCALE_LARGE = (30, 35, 40, 45, 50,)


# Add meta-builder for CompositeGate
class X(CompositeGate):
    def __init__(self, meta_builder: Callable[[CompositeGate], CompositeGate] = lambda x: x):
        super().__init__()
        self.meta_flag: bool = false
        self.meta_builder = meta_builder

    # Maybe overwrite super().gates()
    def unpack(self, prev: CompositeGate) -> Iterable[BasicGate]:
        if self.meta_flag:
            yield from self.gates
        else:
            yield from self.meta_builder(prev)


def parse_scales(scale: str) -> Tuple[int]:
    qubit_scales = tuple()
    if scale == "smale":
        qubit_scales = DEFAULT_SCALE_SMALL
    elif scale == "medium":
        qubit_scales = DEFAULT_SCALE_MEDIUM
    elif scale == "large":
        qubit_scales = DEFAULT_SCALE_LARGE
    else:
        ValueError("Only small/medium/large are allowed to choose circuit scale.")
    return qubit_scales


def default_size(scale: int) -> int:
    return 20 * scale


def random_single_bit_gate() -> BasicGate:
    pass


def random_diag_gate() -> BasicGate:
    pass


def random_ctrl_diag_gate() -> BasicGate:
    pass


def random_unitary_gate() -> BasicGate:
    pass


def random_ctrl_unitary_gate() -> BasicGate:
    pass


class CircuitFactory:
    @staticmethod
    def default(scale: str, gate_generator: Callable[[], BasicGate]) -> Iterable[Circuit]:
        qubit_scales = parse_scales(scale)
        for qubit_num in qubit_scales:
            circ = Circuit(qubit_num)
            for _ in range(default_size(qubit_num)):
                gate_generator() | circ
            yield circ

    @staticmethod
    def qft(scale: str) -> Iterable[Circuit]:
        qubit_scales = parse_scales(scale)
        for qubit_num in qubit_scales:
            circ = Circuit(qubit_num)
            QFT.build_gate(qubit_num) | circ
            yield circ


def single_bit_gate_circuits(scale: str) -> Iterable[Circuit]:
    yield from CircuitFactory.default(scale, random_single_bit_gate)


def diag_gate_circuits(scale: str) -> Iterable[Circuit]:
    yield from CircuitFactory.default(scale, random_diag_gate)


def ctrl_diag_gate_circuits(scale: str) -> Iterable[Circuit]:
    yield from CircuitFactory.default(scale, random_ctrl_diag_gate)


def unitary_gate_circuits(scale: str) -> Iterable[Circuit]:
    yield from CircuitFactory.default(scale, random_unitary_gate)


def ctrl_unitary_gate_circuits(scale: str) -> Iterable[Circuit]:
    yield from CircuitFactory.default(scale, random_ctrl_unitary_gate)


def qft_circuits(scale: str) -> Iterable[Circuit]:
    yield from CircuitFactory.qft(scale)
