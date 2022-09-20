import os
import sys

from QuICT.core import Layout
from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform import USTCSet, GoogleSet, IBMQSet, IonQSet


iset_mapping = {
    "USTC": USTCSet,
    "Google": GoogleSet,
    "IBMQ": IBMQSet,
    "IonQ": IonQSet
}


def qcda_start(
    circuit_path: str,
    output_path: str,
    optimization: str,
    layout_path: str = None,
    instruction_set: str = None
):
    # Get circuit from given path
    circuit = OPENQASMInterface.load_file(circuit_path).circuit

    qcda = QCDA()
    if bool(layout_path):
        layout = Layout.load_file(layout_path)
        qcda.add_default_mapping(layout)

    if bool(optimization):
        qcda.add_default_optimization()

    if instruction_set is not None:
        instruction_set = iset_mapping[instruction_set]
        qcda.add_default_synthesis(instruction_set)

    circuit_opt = qcda.compile(circuit)
    output_path = os.path.join(output_path, 'circuit.qasm')
    circuit_opt.qasm(output_path)


if __name__=="__main__":
    qcda_start(
        *sys.argv[1:]
    )
