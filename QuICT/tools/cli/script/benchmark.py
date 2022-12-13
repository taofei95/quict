import sys
import time
import itertools

from QuICT.core import Circuit
from QuICT.simulation import Simulator
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.tools import Logger, LogFormat
from QuICT.tools.interface import OPENQASMInterface
from QuICT.tools.circuit_library import CircuitLib
from QuICT.qcda.synthesis import GateTransform
from QuICT.qcda.synthesis.gate_transform import USTCSet, IBMQSet, NamSet, GoogleSet
from QuICT.qcda.optimization import (
    CliffordRzOptimization, CommutativeOptimization, SymbolicCliffordOptimization,
    TemplateOptimization, CnotWithoutAncilla
)


logger = Logger("CLI_Benchmark", LogFormat.zero)


def alg_benchmark():
    # Convert qubits from str into List[int]
    alg_list = ["adder", "grover", "maxcut"]
    max_qubits = 10
    circuit_library = CircuitLib()
    simulation = ConstantStateVectorSimulator()

    for alg in alg_list:
        circuits = circuit_library.get_algorithm_circuit(alg, qubits_interval=max_qubits)
        for cir in circuits:
            start_time = time.time()
            _ = simulation.run(cir)
            end_time = time.time()

            logger.info(f"Spending Time: {end_time - start_time}; with {cir.name}")


def qcda_benchmark(circuit_path: str = None):
    if circuit_path == "None":
        circuit = Circuit(10)
        circuit.random_append(200, random_params=True)
    else:
        circuit = OPENQASMInterface.load_file(circuit_path).circuit

    logger.info(f"Based Circuit's Size: {circuit.size()}; Depth: {circuit.depth()}.")
    logger.info("Starting Gate Transform")
    for gset in [USTCSet, IBMQSet, NamSet, GoogleSet]:
        temp_cir = GateTransform(gset).execute(circuit)
        logger.info(f"After Gate Transform Circuit's Size: {temp_cir.size()}; Depth: {temp_cir.depth()}.")

    logger.info("Starting Optimization")
    opt_list = [
        CommutativeOptimization, CliffordRzOptimization, SymbolicCliffordOptimization,
        TemplateOptimization, CnotWithoutAncilla
    ]
    for opt in opt_list:
        logger.info(f"Based Circuit's Size: {circuit.size()}; Depth: {circuit.depth()}.")
        try:
            temp_cir = opt().execute(circuit)
            logger.info(f"After Optimization Circuit's Size: {temp_cir.size()}; Depth: {temp_cir.depth()}.")
        except Exception as _:
            logger.info("Failure to use the optimization to the given circuit.")

    # TODO: Add layout test for linear, grid.
    pass


def simulation_benchmark(circuit_path: str = None, gpu_enable: bool = False):
    if circuit_path == "None":
        max_qubits = list(range(10, 20, 4))
        max_qubits_dm = list(range(5, 10, 2))
        circuit_library = CircuitLib()

        circuits = circuit_library.get_random_circuit("ustc", qubits_interval=max_qubits)
        circuits_dm = circuit_library.get_random_circuit("ustc", qubits_interval=max_qubits_dm)
    else:
        circuits = [OPENQASMInterface.load_file(circuit_path).circuit]

    backend_list = ["state_vector", "density_matrix", "unitary"]
    device_list = ["CPU"]
    if gpu_enable:
        device_list.append("GPU")

    for backend, device in itertools.product(backend_list, device_list):
        simulator = Simulator(device, backend)
        if backend == "state_vector" or circuit_path is not None:
            circuits_list = circuits
        else:
            circuits_list = circuits_dm

        for cir in circuits_list:
            start_time = time.time()
            _ = simulator.run(cir)
            end_time = time.time()

            logger.info(f"Spending Time: {end_time - start_time}; with {cir.name} and backend {backend}, {device}")


if __name__ == "__main__":
    type_ = sys.argv[1]
    
    if type_ == "algorithm":
        alg_benchmark()
    elif type_ == "qcda":
        qcda_benchmark(sys.argv[2])
    else:
        simulation_benchmark(*sys.argv[2:])
