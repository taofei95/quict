from circuit_gen import *


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


if __name__ == "__main__":
    from os import path, makedirs

    result_dir_name = "circ_qasm"
    if not path.isdir(result_dir_name):
        makedirs(result_dir_name)
    method_list = [method for method in dir(Benchmarks) if not method.startswith("__")]
    scale_list = ["small", "medium", "large"]
    for scale in scale_list:
        scaled_path = path.join(result_dir_name, scale)
        if not path.isdir(scaled_path):
            makedirs(scaled_path)
        for method_name in method_list:
            print(f"Generating for {scale} {method_name}...")
            method = getattr(Benchmarks, method_name)
            circ_dir_path = path.join(scaled_path, method_name)
            if not path.isdir(circ_dir_path):
                makedirs(circ_dir_path)
            for idx, circ in enumerate(method(scale)):
                circ: Circuit
                qasm_path = path.join(circ_dir_path, f"{idx}.qasm")
                with open(qasm_path, "w") as f:
                    f.write(circ.qasm())
