import argparse
from distutils.log import info
import fcntl
import json
import os
import platform
import shlex
import struct
import termios
import time
from tkinter.messagebox import NO
import traceback
from werkzeug.utils import secure_filename
from flask_cors import CORS
from loguru import logger
from QuICT.qcda.qcda import QCDA
import uuid
import numpy as np
import cupy
import math
import cmath
from pathlib import Path

from flask import Flask, send_file, send_from_directory, flash, request, redirect, url_for, make_response, abort, render_template, session
from flask_socketio import SocketIO, send, disconnect, emit
from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.synthesis.gate_transform.instruction_set import InstructionSet
from QuICT.tools.interface import OPENQASMInterface
# from QuICT.qcda.simulation.statevector_simulator import ConstantStateVectorSimulator
from QuICT.simulation.simulator import Simulator
from QuICT.lib import Qasm
# from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization
from QuICT.qcda.synthesis.gate_transform import *
import functools
import flask_login
from flask_login import current_user, login_required

UPLOAD_FOLDER = './temp'
ALLOWED_EXTENSIONS = {'txt', 'log', 'qasm'}
port = 5000

app = Flask("vp-qcda")
app.config["fd"] = None
app.config["pid"] = None
app.config["child_pid"] = None
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# LOGIN PART
app.secret_key = b'ivTlmlrj8GEV+msF9DDRWU9LFtT'

login_manager = flask_login.LoginManager()

login_manager.init_app(app)

users = {'ICT_Quantum': {'password': 'Algorithm'}}


class User(flask_login.UserMixin):
    pass


@login_manager.user_loader
def user_loader(email):
    if email not in users:
        return

    user = User()
    user.id = email
    return user


@login_manager.request_loader
def request_loader(request):
    email = request.form.get('email')
    if email not in users:
        return

    user = User()
    user.id = email
    return user


def authenticated_only(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        return f(*args, **kwargs)  # for develop
        if not current_user.is_authenticated:
            disconnect()
        else:
            return f(*args, **kwargs)
    return wrapped


# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'GET':
#         return '''
#                <form action='login' method='POST'>
#                 <input type='text' name='email' id='email' placeholder='email'/>
#                 <input type='password' name='password' id='password' placeholder='password'/>
#                 <input type='submit' name='submit'/>
#                </form>
#                '''

#     email = request.form['email']
#     if request.form['password'] == users[email]['password']:
#         user = User()
#         user.id = email
#         flask_login.login_user(user)
#         return redirect(url_for('index'))

#     return 'Bad login'


@app.route('/logout')
def logout():
    flask_login.logout_user()
    return 'Logged out'


@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect(url_for('index'))


# WEB BASIC PART

@app.route("/")
# @login_required
def index():
    return send_file("./dist/index.html")


@app.route("/<path:path>")
@login_required
def root_folder(path: str):
    return send_from_directory("./dist/", path)


@app.route("/assets/<path:path>")
# @login_required
def assets_files(path: str):
    return send_from_directory("./dist/assets", path)


@app.route('/api/download_file', methods=['GET', 'POST'])
def download_file():
    file_path = request.values['name']
    logger.info(file_path)
    if os.path.exists(file_path):
        return send_file(file_path)


# SOCKET IO BASIC PART

socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("connect", namespace="/api/pty")
@authenticated_only
def connect():
    """new client connected"""
    pass

# Login PART

@socketio.on("login", namespace="/api/pty")
def login(content):
    uid = content['uuid']
    content = content['content']
    usr = content['user']
    psw = content['psw']
    if psw == users[usr]['password']:
        user = User()
        user.id = usr
        flask_login.login_user(user)
        emit('login_success', {'uuid': uid,}, namespace="/api/pty")
    
@socketio.on("testLogin", namespace="/api/pty")
def testLogin(content):
    uid = content['uuid']

    if flask_login.current_user is not None and flask_login.current_user.is_authenticated:
        emit('login_success', {'uuid': uid,}, namespace="/api/pty")
    else:
        emit('need_login', {'uuid': uid,}, namespace="/api/pty")

# QCDA PART

@socketio.on("get_gate_set", namespace="/api/pty")
@authenticated_only
def get_gate_set(content):
    uid = content['uuid']
    source = content['source']    
    gate_set = []
    # QFT, IQFT,Perm, ControlPermMulDetail, PermShift, ControlPermShift, PermMul, ControlPermMul, PermFx, Unitary, ShorInitial,
    FullSet = [H, S, S_dagger, X, Y, Z, SX, SY, SW, ID, U1, U2, U3, Rx, Ry, Rz, T, T_dagger, Phase, CZ,
               CX, CY, CH, CRz, CU1, CU3, FSim, Rxx, Ryy, Rzz, Measure, Reset, Barrier, Swap, CCX, CCRz,  CSwap]
    for name, gateset in {'FullSet': FullSet, 'GoogleSet': [FSim, SX, SY, SW, Rx, Ry], 'IBMQSet': [CX, Rz, SX, X], 'IonQSet': [Rxx, Rx, Ry, Rz], 'USTCSet': [CX, Rx, Ry, Rz, H, X]}.items():
        logger.warning(name)
        simpleSet = {'name': name, 'gates': []}
        for gate in gateset:

            try:
                matrix = np.array2string(gate.compute_matrix, separator=',\t')
            except Exception:
                matrix = None

            # logger.info(matrix)

            pi_args = []
            for arg in gate.pargs:
                if arg == np.pi/2:
                    pi_args.append('pi/2')
                elif arg == -np.pi/2:
                    pi_args.append('-pi/2')
                else:
                    pi_args.append(str(arg))

            simpleSet['gates'].append({
                "targets": gate.targets,
                "controls": gate.controls,
                "name": gate.__class__.__name__,
                "pargs": pi_args,
                "img": f"{gate.__class__.__name__}.png",
                "matrix": matrix,
                "qasm_name": str(gate.qasm_name).lower(),
            })
        gate_set.append(simpleSet)
    if source == 'QCDA':
        emit('n_all_sets', {'uuid': uid,
                        'all_sets': gate_set}, namespace="/api/pty")
    elif source == 'QCDA_load':
        emit('l_all_sets', {'uuid': uid,
                        'all_sets': gate_set}, namespace="/api/pty")
    else:
        emit('all_sets', {'uuid': uid,
                        'all_sets': gate_set}, namespace="/api/pty")


@socketio.on("qasm_load", namespace="/api/pty")
@authenticated_only
def load_file(content):
    uid = content['uuid']
    source = content['source']
    program_text = content['content']
    
    try:
        circuit = load_data(data=program_text)
    except Exception as e:
        if source == 'QCDA':
            emit(
                'QCDA_info', {'uuid': uid, 'info': f"Illegal Qasm file: {e}."}, namespace="/api/pty")
        else:
            emit(
                'info', {'uuid': uid, 'info': f"Illegal Qasm file: {e}."}, namespace="/api/pty")
        logger.warning(
            f"error parse qasm file {e} {traceback.format_exc()}")
        return
    if source == 'QCDA':
        # optimize 
        optimize = content['optimize']
        mapping = content['mapping']
        topology = content['topology']
        set = content['set']
        logger.info(f'circuit.qasm(): {circuit.qasm()}')
        optimized_q = optimize_qasm(uid=uid, qasm_text=program_text, topology=topology, set=set, optimize=optimize, mapping=mapping)
        
        emit(
            "QCDA_o_qasm_load", {'uuid': uid, 'qasm': optimized_q.qasm()}, namespace="/api/pty")
        
        gates = get_gates_list(optimized_q)
        emit('QCDA_o_gates_update', {'uuid': uid, 'gates': gates}, namespace="/api/pty")
    else:
        # no optimize
        emit(
            "qasm_load", {'uuid': uid, 'qasm': circuit.qasm()}, namespace="/api/pty")

        gates = get_gates_list(circuit)
        emit('gates_update', {'uuid': uid, 'gates': gates}, namespace="/api/pty")

        

def optimize_qasm(uid, qasm_text, topology, set, optimize, mapping): 

    circuit = load_data(data=qasm_text)

    circuit_topology = Layout(circuit.width())
    for edge in topology:
        uv = edge.split('_')
        u = int(uv[0])
        v = int(uv[1])
        circuit_topology.add_edge(u, v)

    if set['name'] not in ['FullSet', 'CustomerSet']:
        circuit_set = eval(set['name'])
    else:
        gate_set2 = []
        gate_set1 = []
        for gate_str in set['gates']:
            t_gate = eval(gate_str['name'])()
            if t_gate._controls + t_gate._targets==1:
                gate_set1.append(t_gate._type)
            else :
                # gate_set2.append(t_gate._type)
                gate_set2 = t_gate._type
        circuit_set = InstructionSet(gate_set2, gate_set1)
    

    emit(
        'info',  {'uuid': uid, 'info': f"Compiling circuit..."}, namespace="/api/pty")

    qcda = QCDA()
    qcda.add_default_synthesis(circuit_set)
    if optimize:
        qcda.add_default_optimization()
    if mapping:
        qcda.add_default_mapping(circuit_topology)

    qcda.add_default_synthesis(circuit_set)
    circuit_phy = qcda.compile(circuit)
    
    logger.info(f"circuit_phy.qasm() {circuit_phy.qasm()}")

    # instance = OPENQASMInterface()
    # instance.load_circuit(circuit_phy)
    
    return circuit_phy

@socketio.on("qasm_save", namespace="/api/pty")
@authenticated_only
def save_file(content):
    uid = content['uuid']
    content = content['content']
    filename = str(uuid.uuid1()) + ".qasm"
    filename = secure_filename(filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(file_dir):
        Path(file_dir).mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)
    emit('download_uri', {'uuid': uid, 'download_uri': url_for(
        'download_file', name=file_path)}, namespace="/api/pty")


@socketio.on("qasm_run", namespace="/api/pty")
@authenticated_only
def run_file(content):
    uid = content['uuid']
    data = content['content']
    optimize = content['optimize']
    mapping = content['mapping']
    topology = content['topology']
    set = content['set']
    setting = content['setting']
    logger.info(f"run content {content}")
    try:

        circuit_phy = optimize_qasm(uid=uid, qasm_text=data, topology=topology, set=set, optimize=optimize, mapping=mapping)
        circuit = circuit_phy

        logger.info(f"run qasm {circuit.qasm()}")
        emit(
            'info',  {'uuid': uid, 'info': f"Running circuit..."}, namespace="/api/pty")
        simulation = Simulator(**setting)
        result = simulation.run(circuit)
        emit(
            'info', {'uuid': uid, 'info': f"Run circuit finished."}, namespace="/api/pty")
        emit('run_result', {'uuid': uid, 'run_result': result}, namespace="/api/pty")
    except Exception as e:
        import traceback
        logger.warning(f"Run circuit error: {e}, {traceback.format_exc()}")
        emit(
            'info', {'uuid': uid, 'info': f"Run circuit error: {e}"}, namespace="/api/pty")

@socketio.on("o_qasm_run", namespace="/api/pty")
@authenticated_only
def o_run_file(content):
    uid = content['uuid']
    data = content['content']
    setting = content['setting']
    logger.info(f"run content {content}")
    try:

        circuit = load_data(data=data)
        logger.info(f"run qasm {circuit.qasm()}")
        emit(
            'info',  {'uuid': uid, 'info': f"Running circuit..."}, namespace="/api/pty")
        simulation = Simulator(**setting)
        result = simulation.run(circuit)
        emit(
            'info', {'uuid': uid, 'info': f"Run circuit finished."}, namespace="/api/pty")
        emit('o_run_result', {'uuid': uid, 'run_result': result}, namespace="/api/pty")
    except Exception as e:
        import traceback
        logger.warning(f"Run circuit error: {e}, {traceback.format_exc()}")
        emit(
            'info', {'uuid': uid, 'info': f"Run circuit error: {e}"}, namespace="/api/pty")

def cal_mod(x, y):
    z = complex(x, y)
    return cmath.polar(z)

def load_data(data) -> OPENQASMInterface:
    instance = OPENQASMInterface()
    instance.ast = Qasm(data=data).parse()
    instance.analyse_circuit_from_ast(instance.ast)

    return instance.circuit


@socketio.on("programe_update", namespace="/api/pty")
@authenticated_only
def programe_update(content):
    uid = content['uuid']
    ProgramText = content['content']
    try:
        emit(
            'info',  {'uuid': uid, 'info': f"updating gates..."}, namespace="/api/pty")
        qasm = load_data(data=ProgramText)
        gates = get_gates_list(qasm)

        emit('gates_update', {'uuid': uid,
             'gates': gates}, namespace="/api/pty")
        emit(
            'info', {'uuid': uid, 'info': f"gates updated."}, namespace="/api/pty")
    except Exception as e:
        import traceback
        logger.warning(f"update gates error: {e}, {traceback.format_exc()}")
        emit(
            'info', {'uuid': uid, 'info': f"update gates error: {e}"}, namespace="/api/pty")


def get_gates_list(qasm:Circuit):
    gates_org = qasm.gates
    gates = []

    for gate in gates_org:
        try:
            matrix = np.array2string(gate.compute_matrix, separator=',\t')
        except Exception:
            matrix = None

        logger.info(matrix)

        pi_args = []
        for arg in gate.pargs:
            if arg == np.pi/2:
                pi_args.append('pi/2')
            elif arg == -np.pi/2:
                pi_args.append('-pi/2')
            else:
                pi_args.append(str(arg))

        gates.append({
            "targets": gate.targs,
            "controls": gate.cargs,
            "name": gate.__class__.__name__,
            "pargs": pi_args,
            "img": f"{gate.__class__.__name__}.png",
            "matrix": matrix,
            "qasm_name": str(gate.qasm_name).lower(),
        })
    return gates


# SERVER BASIC PART

cors = CORS(app, resources={
            r"/api/*": {"origins": "*"}, r"/socket.io/*": {"origins": "*"}})


def os_is_windows() -> bool:
    info = platform.platform()
    if "Windows" in info:
        return True
    else:
        return False


def shell_cmd() -> str:
    if os_is_windows():
        return "cmd.exe"
    else:
        return "bash"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Welcome to VP-QCDA. "
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--port", default=port,
                        help="port to run server on")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="host to run server on (use 0.0.0.0 to allow access from other hosts)",
    )
    parser.add_argument("--debug", action="store_true",
                        help="debug the server")
    parser.add_argument("--version", action="store_true",
                        help="print version and exit")
    parser.add_argument(
        "--command", default=shell_cmd(), help="Command to run in the terminal"
    )
    parser.add_argument(
        "--cmd-args",
        default="",
        help="arguments to pass to command (i.e. --cmd-args='arg1 arg2 --flag')",
    )
    args = parser.parse_args()
    print(f"serving on http://{args.host}:{args.port}")
    app.config["cmd"] = [args.command] + shlex.split(args.cmd_args)
    socketio.run(app, debug=args.debug, port=args.port, host=args.host)


if __name__ == "__main__":
    main()