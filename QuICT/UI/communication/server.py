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

from common.utils.email_sender import send_reset_password_email
from common.script.redis_controller import RedisController
from common.script.sql_controller import SQLManger
from common.utils.file_manage import create_user_folder
from common.utils.get_config import get_default_user_config

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

@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect(url_for('index'))


# WEB BASIC PART

@app.route("/")
# @login_required
def index():
    return send_file("./dist/index.html")


@app.route("/<path:path>")
# @login_required
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
    if SQLManger().validate_user(usr) :

        if SQLManger().validation_password(usr, psw) : #psw == users[usr]['password']:
            user = User()
            user.id = usr
            userinfo = SQLManger().get_user_info(usr)
            flask_login.login_user(user)
            emit('login_success', {'uuid': uid, 'info':userinfo}, namespace="/api/pty")
        else:
            emit('login_error', {'uuid': uid,}, namespace="/api/pty")

    else:
        emit('error_msg', {'uuid': uid, 'error_msg': 'Failed to Login, enter the correct User.'})
    
@socketio.on("testLogin", namespace="/api/pty")
def testLogin(content):
    uid = content['uuid']

    if flask_login.current_user is not None and flask_login.current_user.is_authenticated:
        emit('login_success', {'uuid': uid,}, namespace="/api/pty")
    else:
        emit('need_login', {'uuid': uid,}, namespace="/api/pty")

@socketio.on("logout", namespace="/api/pty")
def logout(content):
    flask_login.logout_user()
    return 'Logged out'

@socketio.on("register", namespace="/api/pty")
@authenticated_only
def register(content):
    uid = content['uuid']
    content = content['content']
    usr = content['user']
    psw = content['psw']
    email = content['email']
    json_dict = {
            'username': usr,
            'password': psw,
            'email': email,
            'level': 0,
            }

    # Create user folder
    try:
        if not SQLManger().validate_user(usr) :

            create_user_folder(usr)

            # Update user info for SQL and Redis
            SQLManger().add_user(json_dict)
            # RedisController().update_user_dynamic_info(usr, get_default_user_config(usr))
            emit('register_ok', {'uuid': uid,}, namespace="/api/pty")

        else:
            emit('error_msg', {'uuid': uid, 'error_msg': 'User existed.'})

    except:
        emit('error_msg', {'uuid': uid, 'error_msg': 'Failed to register.'})

@socketio.on("unsubscribe", namespace="/api/pty")
@authenticated_only
def unsubscribe(content):
    """ Delete an user. """
    uid = content['uuid']
    content = content['content']
    usr = content['user']
    # redis_controller = RedisController()
    # job_list = redis_controller.list_jobs(username, name_only=True)
    # for job_name in job_list:
    #     redis_controller.add_operator(job_name, JobOperatorType.delete)

    # # Delete user in Redis, need to wait all jobs delete first.
    # redis_controller.add_operator(username, JobOperatorType.user_delete)

    # Delete user information in database
    SQLManger().delete_user(usr)
    emit('unsubscribe_ok', {'uuid': uid,}, namespace="/api/pty")

@socketio.on("forget", namespace="/api/pty")
@authenticated_only
def forget_password(content):
    uid = content['uuid']
    content = content['content']
    usr = content['user']
    email = content['email']
    """ Send email for user for activate new password. """
    if not SQLManger().validate_user(usr) :
        user_info = SQLManger().get_user_info(usr)
        user_email = user_info[1]
        if user_email != email:
            emit('error_msg', {'uuid': uid, 'error_msg': 'Email not correct.'})
        else:
            # Send email to user
            reset_password = send_reset_password_email(user_email)
            SQLManger().update_password(usr, reset_password)

            emit('forget_ok', {'uuid': uid,}, namespace="/api/pty")
    else:
        emit('error_msg', {'uuid': uid, 'error_msg': 'User not existed.'})

@socketio.on("changepsw", namespace="/api/pty")
@authenticated_only    
def update_password(content):
    uid = content['uuid']
    content = content['content']
    usr = content['user']
    old_password = content['old_password']
    new_password = content['new_password']
    if SQLManger().validation_password(usr, old_password):
        """ Update user's password. """
        SQLManger().update_password(usr, new_password)

        emit('update_psw_ok', {'uuid': uid,}, namespace="/api/pty")
    else:
        emit('error_msg', {'uuid': uid, 'error_msg': 'Old Password Not Correct.'})

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
    for name, gateset in {'FullSet': FullSet, 'CustomerSet':[], 'GoogleSet': [FSim, SX, SY, SW, Rx, Ry], 'IBMQSet': [CX, Rz, SX, X], 'IonQSet': [Rxx, Rx, Ry, Rz], 'USTCSet': [CX, Rx, Ry, Rz, H, X]}.items():
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
                'qcda_info', {'uuid': uid, 'info': f"Illegal Qasm file: {e}."}, namespace="/api/pty")
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
        org_q = load_data(data=program_text)
        org_gates = get_gates_list(org_q)
        emit(
            "QCDA_o_qasm_load", {'uuid': uid, 'qasm': optimized_q.qasm()}, namespace="/api/pty")
        logger.info(f'optimized_q: {optimized_q}')
        gates = get_gates_list(optimized_q)
        logger.info(f'gates: {gates}')
        emit('QCDA_o_gates_update', {'uuid': uid, 'gates': gates, 'gates2':org_gates}, namespace="/api/pty")
    else:
        # no optimize
        emit(
            "qasm_load", {'uuid': uid, 'qasm': circuit.qasm()}, namespace="/api/pty")

        gates = get_gates_list(circuit)
        emit('gates_update', {'uuid': uid, 'gates': gates}, namespace="/api/pty")

        

def optimize_qasm(uid, qasm_text, topology, set, optimize, mapping): 
    circuit = load_data(data=qasm_text)
    if not optimize and not mapping:
        return circuit
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
        circuit_set.register_one_qubit_rule(xyx_rule)
    

    emit(
        'info',  {'uuid': uid, 'info': f"Compiling circuit..."}, namespace="/api/pty")

    qcda = QCDA()
    if set['name'] not in ['FullSet']:
        qcda.add_default_synthesis(circuit_set)
    if optimize:
        qcda.add_default_optimization()
    if mapping:
        qcda.add_default_mapping(circuit_topology)

    if set['name'] not in ['FullSet']:
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

class NumpyEncoder(json.JSONEncoder):
    
    def default(self, myobj):
        if isinstance(myobj, np.integer):
            return int(myobj)
        elif isinstance(myobj, np.floating):
            return float(myobj)
        elif isinstance(myobj, np.ndarray):
            return myobj.tolist()
        return json.JSONEncoder.default(self, myobj)

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
    logger.info(f"run content {content} \nusing setting {setting}")
    try:

        circuit = optimize_qasm(uid=uid, qasm_text=data, topology=topology, set=set, optimize=optimize, mapping=mapping)

        logger.info(f"run qasm {circuit.qasm()}")
        emit(
            'info',  {'uuid': uid, 'info': f"Running circuit..."}, namespace="/api/pty")
        simulation = Simulator(**setting)
        result = simulation.run(circuit)
        emit(
            'info', {'uuid': uid, 'info': f"Run circuit finished."}, namespace="/api/pty")
        logger.info(f"run result {result}")
        if result["data"]["state_vector"] is not None:
            state = result["data"]["state_vector"] 

            state_np = cupy.asnumpy(state)
            state_np_r = np.real(state_np)
            state_np_i = np.imag(state_np)
            state_str_r = np.array2string(state_np_r, separator=',', formatter={
                                        'float_kind': lambda x: "\""+("%f" % x).rstrip('0').rstrip('.')+"\""})
            state_str_i = np.array2string(state_np_i, separator=',', formatter={
                                        'float_kind': lambda x: "\""+("%f" % x).rstrip('0').rstrip('.')+"\""})
            # logger.info( state_str_r, state_str_i)
            state_r = json.loads(state_str_r)
            state_i = json.loads(state_str_i)
            index = []
            state_amp = []
            state_ang = []
            for i in range(len(state_r)):
                index.append(format(i, f'0{int(math.log(len(state_r),2))}b'))
                amp, ang = cal_mod(state_np_r[i], state_np_i[i])
                state_amp.append(amp)
                state_ang.append(ang)

            result["data"]["state_vector"]  = list(
                zip(index, state_r, state_i, state_amp, state_ang))

        if result["data"]["density_matrix"] is not None:
            state = result["data"]["density_matrix"] 

            state_np = state # cupy.asnumpy(state)
            state_np_r = np.real(state_np)
            state_np_i = np.imag(state_np)
            
            # state_str_r = np.array2string(state_np_r, separator=',', formatter={
            #                             'float_kind': lambda x: "\""+("%f" % x).rstrip('0').rstrip('.')+"\""})
            # state_str_i = np.array2string(state_np_i, separator=',', formatter={
            #                             'float_kind': lambda x: "\""+("%f" % x).rstrip('0').rstrip('.')+"\""})

            state_str_r = json.dumps(state_np_r, cls=NumpyEncoder)
            state_str_i = json.dumps(state_np_i, cls=NumpyEncoder)

            logger.info(f"state_np_r{state_str_r}")
            logger.info(f"state_np_i{state_str_i}")
            # logger.info( state_str_r, state_str_i)
            state_r = json.loads(state_str_r)
            state_i = json.loads(state_str_i)
            row_0 = ['']
            rows = [row_0]

            for i in range(len(state_r)):
                i2bin = format(i, f'0{int(math.log(len(state_r),2))}b')
                row_0.append(i2bin)
                t_row = [i2bin]
                for j in range(len(state_r)):
                    t_row.append(f'{state_r[i][j]}{"+"if state_i[i][j] >=0 else "-"}{state_i[i][j]}j')

                rows.append(t_row)

            result["data"]["density_matrix"]  = rows

            logger.info(f"rows{rows}")

        emit('run_result', {'uuid': uid, 'run_result': result}, namespace="/api/pty")
    except Exception as e:
        import traceback
        logger.warning(f"Run circuit error: {e}, {traceback.format_exc()}")
        emit(
            'info', {'uuid': uid, 'info': f"Run circuit error"}, namespace="/api/pty")

@socketio.on("o_qasm_run", namespace="/api/pty")
@authenticated_only
def o_run_file(content):
    uid = content['uuid']
    data = content['content']
    optimize = content['optimize']
    mapping = content['mapping']
    topology = content['topology']
    set = content['set']
    setting = content['setting']
    logger.info(f"run content {content}")
    try:
         
        circuit = optimize_qasm(uid=uid, qasm_text=data, topology=topology, set=set, optimize=optimize, mapping=mapping)
        # circuit = load_data(data=data)
        logger.info(f"run qasm {circuit.qasm()}")
        emit(
            'info',  {'uuid': uid, 'info': f"Running circuit..."}, namespace="/api/pty")
        simulation = Simulator(**setting)
        result = simulation.run(circuit)
        emit(
            'info', {'uuid': uid, 'info': f"Run circuit finished."}, namespace="/api/pty")
        logger.info(f"run result {result}")
        if result["data"]["state_vector"] is not None:
            state = result["data"]["state_vector"] 

            state_np = cupy.asnumpy(state)
            state_np_r = np.real(state_np)
            state_np_i = np.imag(state_np)
            state_str_r = np.array2string(state_np_r, separator=',', formatter={
                                        'float_kind': lambda x: "\""+("%f" % x).rstrip('0').rstrip('.')+"\""})
            state_str_i = np.array2string(state_np_i, separator=',', formatter={
                                        'float_kind': lambda x: "\""+("%f" % x).rstrip('0').rstrip('.')+"\""})
            # logger.info( state_str_r, state_str_i)
            state_r = json.loads(state_str_r)
            state_i = json.loads(state_str_i)
            index = []
            state_amp = []
            state_ang = []
            for i in range(len(state_r)):
                index.append(format(i, f'0{int(math.log(len(state_r),2))}b'))
                amp, ang = cal_mod(state_np_r[i], state_np_i[i])
                state_amp.append(amp)
                state_ang.append(ang)

            result["data"]["state_vector"]  = list(
                zip(index, state_r, state_i, state_amp, state_ang))

        if result["data"]["density_matrix"] is not None:
            state = result["data"]["density_matrix"] 

            state_np = state # cupy.asnumpy(state)
            state_np_r = np.real(state_np)
            state_np_i = np.imag(state_np)
            
            # state_str_r = np.array2string(state_np_r, separator=',', formatter={
            #                             'float_kind': lambda x: "\""+("%f" % x).rstrip('0').rstrip('.')+"\""})
            # state_str_i = np.array2string(state_np_i, separator=',', formatter={
            #                             'float_kind': lambda x: "\""+("%f" % x).rstrip('0').rstrip('.')+"\""})

            state_str_r = json.dumps(state_np_r, cls=NumpyEncoder)
            state_str_i = json.dumps(state_np_i, cls=NumpyEncoder)

            logger.info(f"state_np_r{state_str_r}")
            logger.info(f"state_np_i{state_str_i}")
            # logger.info( state_str_r, state_str_i)
            state_r = json.loads(state_str_r)
            state_i = json.loads(state_str_i)
            row_0 = ['']
            rows = [row_0]

            for i in range(len(state_r)):
                i2bin = format(i, f'0{int(math.log(len(state_r),2))}b')
                row_0.append(i2bin)
                t_row = [i2bin]
                for j in range(len(state_r)):
                    t_row.append(f'{state_r[i][j]}{"+"if state_i[i][j] >=0 else "-"}{state_i[i][j]}j')

                rows.append(t_row)

            result["data"]["density_matrix"]  = rows

            logger.info(f"rows{rows}")
        emit('o_run_result', {'uuid': uid, 'run_result': result}, namespace="/api/pty")
    except Exception as e:
        import traceback
        logger.warning(f"Run circuit error: {e}, {traceback.format_exc()}")
        emit(
            'info', {'uuid': uid, 'info': f"Run circuit error: {e}"}, namespace="/api/pty")

def cal_mod(x, y):
    z = complex(x, y)
    return cmath.polar(z)

def load_data(data) -> Circuit:
    instance = OPENQASMInterface()
    instance.ast = Qasm(data=data).parse()
    instance.analyse_circuit_from_ast(instance.ast)

    return instance.circuit

@socketio.on("qcda_load", namespace="/api/pty")
@authenticated_only
def qcda_load(content):
    logger.info(f"qcda_load: {content}")
    uid = content['uuid']
    ProgramText = content['content']
    try:
        emit(
            'qcda_info',  {'uuid': uid, 'info': f"loading gates..."}, namespace="/api/pty")
        qasm = load_data(data=ProgramText)
        gates = get_gates_list(qasm)

        emit('gates_load', {'uuid': uid,
             'gates': gates}, namespace="/api/pty")
        emit(
            'qcda_info', {'uuid': uid, 'info': f"gates loaded."}, namespace="/api/pty")
    except Exception as e:
        import traceback
        logger.warning(f"load gates error: {e}, {traceback.format_exc()}")
        emit(
            'qcda_info', {'uuid': uid, 'info': f"load gates error: {e}"}, namespace="/api/pty")

@socketio.on("programe_update", namespace="/api/pty")
@authenticated_only
def programe_update(content):
    logger.info(f"programe_update: {content}")
    uid = content['uuid']
    ProgramText = content['content']
    source = content['source']
    try:
        emit(
            'info',  {'uuid': uid, 'info': f"updating gates..."}, namespace="/api/pty")
        qasm = load_data(data=ProgramText)
        gates = get_gates_list(qasm)
        if source == 'QCDA':
            emit('QCDA_gates_update', {'uuid': uid,
             'gates': gates}, namespace="/api/pty")
        else:
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

        logger.info(gate)

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
