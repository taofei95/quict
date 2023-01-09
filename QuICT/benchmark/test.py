import math
import random
import re

import scipy.stats
from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
import numpy as np
import os
from QuICT.tools.circuit_library.circuitlib import CircuitLib

from QuICT.tools.circuit_library.get_benchmark_circuit import BenchmarkCircuitBuilder

########################################################################################
def sim(cir):
    result = ConstantStateVectorSimulator().run(cir).get()
    return result
bench = QuICTBenchmark()
# layout = Layout.load_file(os.path.dirname(os.path.abspath(__file__)) + f"/data/layout/line{circuit.width()}.layout")
# Inset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])   
cirs = bench.run(simulator_interface=sim, quantum_machine_info=[5], level=3)
# cirs = bench.get_circuits(quantum_machine_info=[5, Inset], level=2, gate_transform=True)
# for i in cirs:
#     print(i.name)
##########################################################################################


# files=os.listdir("test_results/1")
# files.sort()
# print(len(cirs))
    # results_list.append(np.load(str(file)))
# print(results_list)

# for a in cirs:
#     sim = ConstantStateVectorSimulator()
#     result = sim.run(a).get()
#     np.save(f"{a.name}.npy",result)

###########################################################
# ben = BenchmarkCircuitBuilder()
# cir = ben.entangled_circuit_build(5, 20)
# print(cir[1])
# # for c in cir:
# #     print(c.depth())

# cir[0][0].draw(filename="0")
# cir[0][1].draw(filename="1")
# cir[0][2].draw(filename="2")
# cir[0][3].draw(filename="3")
############################################################

############################################################
# lib = CircuitLib()
# cirs_list = []
# based_fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"]
# for field in based_fields_list:
#     cirs = lib.get_benchmark_circuit(str(field), 5, 10)
#     cirs_list.extend(cirs)
# for c in cirs_list:
#     print(c.name)
###############################################################