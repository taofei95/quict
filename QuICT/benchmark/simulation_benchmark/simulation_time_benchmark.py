import time
from QuICT.core.circuit.circuit import Circuit
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulation
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
from QuICT.simulation.unitary.unitary_simulator import UnitarySimulator

cpu_simulator_list = [CircuitSimulator(), DensityMatrixSimulation(), UnitarySimulator()]
gpu_simulator_list = [ConstantStateVectorSimulator(), DensityMatrixSimulation("GPU"), UnitarySimulator("GPU")]
single_simulator_list = [ConstantStateVectorSimulator("single"), DensityMatrixSimulation(precision="single"), UnitarySimulator(precision="single")]
double_simulator_list = [ConstantStateVectorSimulator(), DensityMatrixSimulation(), UnitarySimulator()]

if __name__ == '__main__':
    data = open('simulation_benchmark_data.txt', 'w+')

    qubit_number = [5, 10]
    for qubit_num in qubit_number:
        data.write(f"qubit number:{qubit_num}, cpu simulation \n")
        circuit = Circuit(qubit_num)
        circuit.random_append(20 * qubit_num)
        for simulator in cpu_simulator_list:
            begin_time = time.time()
            result = simulator.run(circuit)
            last_time = time.time()
            data.write(f"speed:{round(last_time - begin_time, 4)} \n")
        data.write(f"qubit number:{qubit_num}, gpu simulation \n")
        for simulator in gpu_simulator_list:
            begin_time = time.time()
            result = simulator.run(circuit)
            last_time = time.time()
            data.write(f"speed:{round(last_time - begin_time, 4)} \n")
        data.write(f"qubit number:{qubit_num}, single simulation \n")
        for simulator in single_simulator_list:
            begin_time = time.time()
            result = simulator.run(circuit)
            last_time = time.time()
            data.write(f"speed:{round(last_time - begin_time, 4)} \n")
        data.write(f"qubit number:{qubit_num}, double simulation \n")
        for simulator in double_simulator_list:
            begin_time = time.time()
            result = simulator.run(circuit)
            last_time = time.time()
            data.write(f"speed:{round(last_time - begin_time, 4)} \n")

    data.close()



