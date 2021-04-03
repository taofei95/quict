from QuICT.algorithm import *
from QuICT.core import *
from QuICT.qcda.synthesis import HRSIncrementer

circuit = HRSIncrementer(3)
circuit.assign_initial_zeros()
amplitude = Amplitude.run(circuit)
print(amplitude)
