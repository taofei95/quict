from QuICT.simulation.simulator import Simulator
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator

####CPU GPU################

sim_cpu = ConstantStateVectorSimulator(gpu_device_id=2)

sim = Simulator(device="CPU", backend="state_vector", precision="single")    # The path to store result
# result = sim.run(circuits)



#####single double#######