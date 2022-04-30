from .circuits import *


class Benchmarks:
    @classmethod
    def single_bit(cls, scale: str) -> Iterable[Circuit]:
        yield from CircuitFactory.default(scale, random_single_bit_gate)

    @classmethod
    def diag(cls, scale: str) -> Iterable[Circuit]:
        yield from CircuitFactory.default(scale, random_diag_gate)

    @classmethod
    def ctrl_diag(cls, scale: str) -> Iterable[Circuit]:
        yield from CircuitFactory.default(scale, random_ctrl_diag_gate)

    @classmethod
    def unitary(cls, scale: str) -> Iterable[Circuit]:
        yield from CircuitFactory.default(scale, random_unitary_gate)

    @classmethod
    def ctrl_unitary(cls, scale: str) -> Iterable[Circuit]:
        yield from CircuitFactory.default(scale, random_ctrl_unitary_gate)

    @classmethod
    def qft(cls, scale: str) -> Iterable[Circuit]:
        yield from CircuitFactory.qft(scale)
