import numpy as np
from io import TextIOWrapper
from os import path, walk
from typing import Iterable
from QuICT.core.circuit.circuit import Circuit
from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

qasm_parent_dir = path.join(path.dirname(path.abspath(__file__)), "circ_qasm")


def load_circ(scale: str) -> Iterable[str]:
    scaled_qasm_dir = path.join(qasm_parent_dir, scale)
    for root, categories, _ in walk(scaled_qasm_dir):
        for category in categories:
            qasm_dir_path = path.join(root, category)
            for root, _, qasm_files in walk(qasm_dir_path):
                for qasm_file in qasm_files:
                    f_path = path.join(root, qasm_file)
                    yield f_path


def quict_sim(scale: str):
    for f_path in load_circ(scale):
        circ = OPENQASMInterface.load_file(f_path).circuit
        simulator = CircuitSimulator()
        _ = simulator.run(circ)


def qiskit_sim(scale: str):
    # Import Qiskit
    from qiskit import QuantumCircuit
    from qiskit import Aer, transpile
    import qiskit.quantum_info as qi

    for f_path in load_circ(scale):
        print(f"Testing {f_path}")
        circ = QuantumCircuit.from_qasm_file(f_path)

        # Transpile for simulator
        simulator = Aer.get_backend("aer_simulator")
        circ = transpile(circ, simulator)

        # Run and get counts
        result = simulator.run(circ).result()
