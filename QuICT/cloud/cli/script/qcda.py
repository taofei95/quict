import os
import sys

from QuICT.core import Layout
from QuICT.tools import Logger
from QuICT.tools.logger import LogFormat
from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform import USTCSet, GoogleSet, IBMQSet, IonQSet


logger = Logger("QCDA_Local_Mode", LogFormat.full)


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
    logger.debug("Start Run QCDA Job in local mode.")
    logger.debug(
        f"Job Parameters: circuit path: {circuit_path}, optimization: {optimization}, " +
        f"layout path: {layout_path}, Instruction set: {instruction_set}, " +
        f"output path: {output_path}."
    )
    # Get circuit from given path
    circuit = OPENQASMInterface.load_file(circuit_path).circuit

    qcda = QCDA()
    if bool(optimization):
        qcda.add_default_optimization()

    if instruction_set is not None:
        instruction_set = iset_mapping[instruction_set]
        qcda.add_default_synthesis(instruction_set)

    if bool(layout_path):
        layout = Layout.load_file(layout_path)
        qcda.add_default_mapping(layout)

    circuit_opt = qcda.compile(circuit)
    output_path = os.path.join(output_path, 'circuit.qasm')
    circuit_opt.qasm(output_path)

    logger.debug(f"QCDA Job finished, store the result in {output_path}.")


if __name__ == "__main__":
    qcda_start(
        *sys.argv[1:]
    )
