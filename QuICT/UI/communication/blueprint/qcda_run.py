from flask import Blueprint, request
import json

from QuICT.qcda.qcda import QCDA
from QuICT.qcda.simulation.statevector_simulator import ConstantStateVectorSimulator
from QuICT.UI.utils import EncryptCommunication

# Flask related.

blueprint = Blueprint(name="qcda_run", import_name=__name__)
URL_PREFIX = "/qcda"


# Api functions.

@blueprint.route(f"{URL_PREFIX}/opt", methods=["POST"])
def optimizer():
    kwargs = EncryptCommunication.decrypt(request)
    circuit = kwargs["circuit"]
    layout = kwargs["layout"]
    InstructionSet = kwargs["instructionset"]
    optimization = kwargs["optimization"]
    mapping = kwargs["mapping"]

    synthesis = True if InstructionSet != None else False

    if optimization or mapping:
        qcda = QCDA()
        circuit = qcda.compile(
            circuit,
            InstructionSet,
            layout,
            synthesis,
            optimization,
            mapping
        )
    
    return json.dumps(circuit)


@blueprint.route(f"{URL_PREFIX}/run", methods=["POST"])
def run():
    kwargs = EncryptCommunication.decrypt(request)
    circuit = kwargs["circuit"]
    backend = kwargs["backend"]

    # TODO: simulation interface
    simulation = Simulation(backend=backend)
    simulation = ConstantStateVectorSimulator(
        circuit=circuit
    )
    state = simulation.run()

    return json.dumps(circuit.qubits.measured)
