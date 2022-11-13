import random
import numpy as np
import bson


from QuICT.core.circuit.circuit import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.core.gate import *

# get Highly parallelized circuit
def qasm(self, output_file: str = None):
        qreg = self.width()
        creg = min(self.count_gate_by_gatetype(GateType.measure), qreg)
        if creg == 0:
            creg = qreg

        qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm_string += f"qreg q[{qreg}];\n"
        qasm_string += f"creg c[{creg}];\n"

        cbits = 0
        for gate in self._gates:
            if gate.qasm_name == "measure":
                qasm_string += f"measure q[{gate.targ}] -> c[{cbits}];\n"
                cbits += 1
                cbits = cbits % creg
            else:
                qasm_string += gate.qasm()
        
        demoid = bson.ObjectId()

        if output_file is not None:
            with open(output_file, 'w+') as of:
                of.write(qasm_string)
                
        return qasm_string

def Hp_circuit_build(
    qubits: int = 5,
    rand_size: int = 10,
    typelist: list = None,
    random_params: bool = False,
    probabilities: list = None
):
    if typelist is None:
        typelist = [
            GateType.rx, GateType.ry, GateType.rz,
            GateType.cx, GateType.h
        ]

    gate_indexes = list(range(len(typelist)))
    qubits_indexes = list(range(qubits))
    shuffle_qindexes = qubits_indexes[:]
    random.shuffle(shuffle_qindexes)

    cir = Circuit(qubits)
    while cir.size() < rand_size:
        rand_type = np.random.choice(gate_indexes, p=probabilities)
        gate_type = typelist[rand_type]
        gate = GATE_TYPE_TO_CLASS[gate_type]()

        if random_params and gate.params:
            gate.pargs = list(np.random.uniform(0, 2 * np.pi, gate.params))

        gsize = gate.controls + gate.targets
        if gsize > len(shuffle_qindexes):
            continue

        gate & shuffle_qindexes[:gsize] | cir

        if gsize == len(shuffle_qindexes):
            shuffle_qindexes = qubits_indexes[:]
            random.shuffle(shuffle_qindexes)
        else:
            shuffle_qindexes = shuffle_qindexes[gsize:]

    Measure | cir

    cir.draw(filename='parallelized')

    return cir

cir = Hp_circuit_build(5, 40, random_params=True)
print(cir.qasm())

# g2 = cir.count_2qubit_gate()
# P = (cir.size()/cir.depth() -1)/(5 - 1)
# print(P)
# print(cir.size(), cir.depth(), g2)

# f = open("Highly_paralleized_circuit.qasm", 'w+')
# f.write(cir.qasm())

  

