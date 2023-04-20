import json
import re
from curses.ascii import isalpha
from urllib import parse

import requests
from quafu.exceptions import CircuitError, CompileError, ServerError
from quafu.users.exceptions import UserError

from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.tools.interface import OPENQASMInterface

from .backend import Backend

from quafu import User, Task, QuantumCircuit, ExecResult
from quafu.backends.backends import ScQ_P10, Backend as Bkd


class ScQ_P18(Bkd):
    def __init__(self):
        super().__init__("ScQ-P18")
        self.valid_gates = ['cz', 'rx', 'ry', 'rz', 'h']


class ScQ_P136(Bkd):
    def __init__(self):
        super().__init__("ScQ-P136")
        self.valid_gates = ['cz', 'rx', 'ry', 'rz', 'h']


class ModifiedTask(Task):
    def __init__(self):
        super().__init__()

    def send(self,
             qc: QuantumCircuit,
             name: str="",
             group: str="",
            wait: bool=True):
        from quafu import get_version
        version = get_version()
        self.check_valid_gates(qc)
        qc.to_openqasm()
        backends = {"ScQ-P10": 0, "ScQ-P20": 1, "ScQ-P50": 2, "ScQ-S41": 3, "ScQ-P136": 2, "ScQ-P18": 1}
        data = {"qtasm": qc.openqasm, "shots": self.shots, "qubits": qc.num, "scan": 0,
                "tomo": int(self.tomo), "selected_server": backends[self._backend.name],
                "compile": int(self.compile), "priority": self.priority, "task_name": name, "pyquafu_version": version}

        if wait:
            url = self._url + "qbackend/scq_kit/"
        else:
            url = self._url + "qbackend/scq_kit_asyc/"

        headers = {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8', 'api_token': self.token}
        data = parse.urlencode(data)
        data = data.replace("%27", "'")
        res = requests.post(url, headers=headers, data=data)
        res_dict = json.loads(res.text)

        if res.json()["status"] in [201, 205]:
            raise UserError(res_dict["message"])
        elif res.json()["status"] == 5001:
            raise CircuitError(res_dict["message"])
        elif res.json()["status"] == 5003:
            raise ServerError(res_dict["message"])
        elif res.json()["status"] == 5004:
            raise CompileError(res_dict["message"])
        else:
            task_id = res_dict["task_id"]

            if not (group in self.submit_history):
                self.submit_history[group] = [task_id]
            else:
                self.submit_history[group].append(task_id)

            return ExecResult(res_dict, qc.measures)


class QuafuBackend(Backend):
    def __init__(self, api_token, system='ScQ-P10'):
        user = User()
        user.save_apitoken(api_token)
        self.system = system

        task = ModifiedTask()
        task.load_account()
        task.config(backend=system, compile=False)
        if system == 'ScQ-P10':
            task._backend = ScQ_P10()
            n_qubit = 10
        elif system == 'ScQ-P18':
            task._backend = ScQ_P18()
            n_qubit = 18
        elif system == 'ScQ-P136':
            task._backend = ScQ_P136()
            n_qubit = 136
        else:
            assert False, 'unsupported system'

        info = task.get_backend_info()
        mapping = {val: key for key, val in info['mapping'].items()}

        gate_set = []
        gate_str = info['full_info']['basis_gates'].split(',')
        for g in gate_str:
            g = ''.join(filter(lambda x: x.isalpha(), g))
            gate_set.append(GateType.__getattr__(g))

        qubit_t1 = [100] * n_qubit
        qubit_t2 = [100] * n_qubit
        two_qubit_gate_fidelity = {}
        for q in info['full_info']['qubits_info']:
            q_id = mapping[q]
            qubit_t1[q_id] = info['full_info']['qubits_info'][q]['T1']
            qubit_t2[q_id] = info['full_info']['qubits_info'][q]['T2']
        for conn in info['full_info']['topological_structure']:
            pattern = r'(Q\d+)_(Q\d+)'
            q1, q2 = re.findall(pattern, conn)[0]
            q1, q2 = mapping[q1], mapping[q2]
            key = list(info['full_info']['topological_structure'][conn])[0]
            two_qubit_gate_fidelity[(q1, q2)] = info['full_info']['topological_structure'][conn][key]['fidelity']

        super().__init__(
            n_qubit=n_qubit,
            gate_set=gate_set,
            qubit_t1=qubit_t1,
            qubit_t2=qubit_t2,
            two_qubit_gate_fidelity=two_qubit_gate_fidelity
        )

        self.task = task

    def execute_circuit(self, circ: Circuit, n_shot: int, *args, **kwargs) -> list[float]:
        self.task.shots = n_shot
        self.task.compile = False
        if 'compile' in kwargs:
            self.task.compile = kwargs['compile']

        circ_quafu = QuantumCircuit(self.n_qubit)
        circ_quafu.from_openqasm(circ.qasm())
        res = self.task.send(circ_quafu)

        if res.task_status == 'Failed':
            return []

        return res.probabilities
