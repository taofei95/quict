from flask import Blueprint, request
import json

from QuICT.qcda.qcda import QCDA
from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.simulation.statevector_simulator import ConstantStateVectorSimulator
from QuICT.UI.utils import EncryptCommunication


# Flask related.

blueprint = Blueprint(name="composer", import_name=__name__)
URL_PREFIX = "/circuit"

key_dict = {
    "likaiqi": "likaiqi"
}

# Api functions.
@blueprint.route(f"/login", methods=["POST"])
def login():
    username = request.headers.get("username")
    key = key_dict[username]
    kwargs = json.loads(request.data)
    password = kwargs["password"] # qasm string

    if EncryptCommunication.md5_encrypt(key) == password:
        result = True

    result = False

    return json.dumps(result)

@blueprint.route(f"{URL_PREFIX}/run", methods=["POST"])
def run():
    username = request.headers.get("username")
    key = key_dict[username]
    kwargs = EncryptCommunication.decrypt(request, key)
    qasm_string = kwargs["circuit"] # qasm string

    # convert qasm string into Circuit
    qasm = OPENQASMInterface.load_data(data=qasm_string)
    circuit = qasm.circuit
    simulation = ConstantStateVectorSimulator(
        circuit=circuit
    )
    state = simulation.run()

    return EncryptCommunication.base64_encrypt(state.get())
