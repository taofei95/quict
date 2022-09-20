


import os
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


qasm = OPENQASMInterface.load_file(
            os.path.dirname(os.path.abspath(__file__)) + "/../../unit_test/simulation/data/random_circuit_for_correction.qasm"
        )
circuit = qasm.circuit
print(circuit.qasm())
