from typing import List
from QuICT.tools.interface import OPENQASMInterface
from QuICT.core.circuit import * 
from QuICT.core.layout import *
from QuICT.qcda.mapping import Mapping


if __name__ == "__main__": 
    layout = Layout.load_file(f"./ibmq20.layout") 
    qc = OPENQASMInterface.load_file(f"./4gt13_92.qasm").circuit
    transformed_circuit = Mapping.run(circuit = qc, layout = layout)




