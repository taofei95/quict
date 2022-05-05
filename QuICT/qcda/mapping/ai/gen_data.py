import os
from QuICT.core.layout import *
from QuICT.core.gate import *
from QuICT.core.circuit import *
from QuICT.qcda.mapping import MCTSMapping as Mapping
from QuICT.tools.interface import OPENQASMInterface

def gen_data():
    file_path = os.path.realpath(__file__)
    dir, _ = os.path.split(file_path)
    layout = Layout.load_file(f"{dir}/../unittest/ibmq_casablanca.layout")
    # qc = OPENQASMInterface.load_file(f"{dir}/../unittest/example_test.qasm").circuit
    
    for i in range(100):
        qc = Circuit(wires=7)
        qc.random_append(rand_size=30,typelist=[GateType.cx, GateType.h, GateType.x, GateType.y, GateType.z, GateType.t, GateType.tdg])
        transformed_circuit = Mapping.execute(circuit=qc, layout=layout, init_mapping_method="naive")
        with open(f"data/ibmq_casablanca/source/{i}.qasm", "w") as f:
            f.write(qc.qasm())
        with open(f"data/ibmq_casablanca/result/{i}.qasm", "w") as f:
            f.write(transformed_circuit.qasm())
        if 0 == i%10:
            print(f"iter: {i}.")

if __name__ == "__main__":
    gen_data()


