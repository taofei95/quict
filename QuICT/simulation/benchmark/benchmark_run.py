from os import path, walk
from typing import Callable, Iterable, List, Union
from unicodedata import category

from time import time
from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

qasm_parent_dir = path.join(path.dirname(path.abspath(__file__)), "circ_qasm")


def load_circ(scale: str, categories: Union[str, List[str]] = "all") -> Iterable[str]:
    scaled_qasm_dir = path.join(qasm_parent_dir, scale)
    if type(categories) is str and categories.lower() == "all":
        for _, categories, _ in walk(scaled_qasm_dir):
            for category in categories:
                qasm_dir_path = path.join(scaled_qasm_dir, category)
                for _, _, qasm_files in walk(qasm_dir_path):
                    for qasm_file in qasm_files:
                        f_path = path.join(qasm_dir_path, qasm_file)
                        yield f_path
    else:
        if type(categories) is str:
            categories = [categories]
        for category in categories:
            qasm_dir_path = path.join(scaled_qasm_dir, category)
            for root, _, qasm_files in walk(qasm_dir_path):
                for qasm_file in qasm_files:
                    f_path = path.join(root, qasm_file)
                    yield f_path


def quict_sim(scale: str, categories: Union[str, List[str]] = "all"):
    for f_path in load_circ(scale, categories):
        print(f"Testing {f_path}")
        circ = OPENQASMInterface.load_file(f_path).circuit
        simulator = CircuitSimulator()
        _ = simulator.run(circ)


def qiskit_sim(scale: str, categories: Union[str, List[str]] = "all"):
    # Import Qiskit
    from qiskit import QuantumCircuit
    from qiskit import Aer, transpile

    for f_path in load_circ(scale, categories):
        print(f"Testing {f_path}")
        circ = QuantumCircuit.from_qasm_file(f_path)

        # Transpile for simulator
        simulator = Aer.get_backend("aer_simulator")
        circ = transpile(circ, simulator)

        # Run and get counts
        _ = simulator.run(circ).result()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark CLI argument parser.")
    parser.add_argument(
        "--simulator",
        choices=["quict", "qiskit", "all"],
        help="Select the simulator to be used. (Default: all)",
        default="all",
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium", "large"],
        help="Select circuit qubit scales: small(<10), medium(~15), large(>25). (Default: medium)",
        default="medium",
    )
    parser.add_argument(
        "--category",
        choices=[
            "ctrl_diag",
            "ctrl_unitary",
            "diag",
            "qft",
            "single_bit",
            # "unitary",
            "all",
        ],
        help="Select the circuit category to be simulated. (Default: all)",
        default="all",
    )
    args = parser.parse_args()
    simulator: str = args.simulator
    scale: str = args.scale
    category: List[str] = []
    if args.category == "all":
        category = [
            "ctrl_diag",
            "ctrl_unitary",
            "diag",
            "qft",
            "single_bit",
            # "unitary",
        ]
    else:
        category = [category]

    if simulator in ["quict", "all"]:
        print("Testing with QuICT simulator...")
        start_time = time()
        quict_sim(scale, category)
        end_time = time()
        print(f"Finished. Duration: {end_time-start_time:.4f}s")
    if simulator in ["qiskit", "all"]:
        print("Testing with Qiskit simulator...")
        qiskit_sim(scale, category)
        end_time = time()
        print(f"Finished. Duration: {end_time-start_time:.4f}s")
