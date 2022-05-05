from QuICT.core import *
from QuICT.core.gate.gate import *
# from .dag import DAG
from typing import Union

def extract_first_swap(circuit:Circuit) -> Union[None, SwapGate]:
    for gate in circuit.gates:
        if gate.type() == GateType.swap:
            return gate 
    return None
