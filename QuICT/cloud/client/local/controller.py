import os
import redis
import psutil
import multiprocessing as mp

from QuICT.core import Circuit, Layout
from QuICT.simulation import Simulator
from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.qcda import QCDA
from QuICT.qcda.synthesis.gate_transform import *


iset_mapping = {
    "USTC": USTCSet,
    "Google": GoogleSet,
    "IBMQ": IBMQSet,
    "IonQ": IonQSet
}


def simulation_start(circuit: Circuit, simualtor_options: dict, output_path: str):
    simulator = Simulator(
        device=simualtor_options['resource']['device'],
        backend=simualtor_options['backend'],
        shots=simualtor_options['shots'],
        precision=simualtor_options['precision'],
        output_path=output_path
    )
    simulator.run(circuit)


def qcda_start(circuit: Circuit, qcda_options: dict, output_path: str):
    qcda = QCDA()
    map_opt = qcda_options['mapping']
    if map_opt['enable']:
        layout_path = map_opt['layout_path']
        layout = Layout.load_file(layout_path)
        qcda.add_default_mapping(layout)

    opt_opt = qcda_options['optimization']
    if opt_opt['enable']:
        qcda.add_default_optimization()

    sync_opt = qcda_options['synthesis']
    if sync_opt['enable']:
        instruction_set = iset_mapping[sync_opt['instruction_set']]
        qcda.add_default_synthesis(instruction_set)

    circuit_opt = qcda.compile(circuit)
    output_path = os.path.join(output_path, 'circuit.qasm')
    circuit_opt.qasm(output_path)


class QuICTLocalJobManager:
    def __init__(self):
        try:
            self._redis_pool = redis.ConnectionPool(host='127.0.0.1', port=6379, db=0)
            self._redis_connection = redis.Redis(connection_pool=self._redis_pool, decode_responses=True)
        except Exception as e:
            raise KeyError(f"Failure to connect to Redis, due to {e}. please run \'quict local launch\' first.")

    def _name_validation(self, name):
        if not self._redis_connection.exists(name):
            raise KeyError(
                f"Unmatched job name {name} in local jobs, please using \'quict local job list\' first."
            )

        # Update given job's status
        self._update_job_status(name)

    def start_job(self, yml_dict: dict):
        # Check job name
        name = yml_dict["name"]
        if self._redis_connection.exists(name):
            raise KeyError("Repeated name in local jobs.")

        # Get circuit
        cir_path = yml_dict['circuit']
        circuit = OPENQASMInterface.load_file(cir_path).circuit

        # Create Job in job_queue
        job_type = yml_dict['type']
        output_path = yml_dict['output_path']

        # Start job by its purpose
        runtime_args = yml_dict["simulation"] if job_type == "simulation" else yml_dict["qcda"]
        pid = self._start_job(circuit, job_type, output_path, runtime_args)

        # Set job information into Redis.
        job_info = {
            'pid': pid,
            'type': job_type,
            'status': 'running',
            'output_path': output_path
        }
        self._redis_connection.hmset(name, job_info)

    def _start_job(
        self,
        circuit: Circuit,
        job_type: str,
        output_path: str,
        options: dict
    ):
        # Start process for current job
        target_function = simulation_start if job_type == "simulation" else qcda_start
        process = mp.Process(target=target_function, args=(circuit, options, output_path))
        process.start()
        process.join()
        pid = process.pid

        return pid

    def _update_job_status(self, name: str):
        job_status = str(self._redis_connection.hget(name, 'status'), "utf-8")
        if job_status in ["finish", "stop"]:
            return

        job_pid = int(self._redis_connection.hget(name, 'pid'))
        try:
            _ = psutil.Process(job_pid)
            job_status = 'running'   # job_process.status()
        except:
            job_status = "finish"

        self._redis_connection.hset(name, 'status', job_status)

    def status_job(self, name: str):
        # Check job's list contain given job
        self._name_validation(name)

        print(self._redis_connection.hgetall(name))
        return self._redis_connection.hgetall(name)

    def stop_job(self, name: str):
        # Check job's list contain given job
        self._name_validation(name)

        # check job status
        job_status = str(self._redis_connection.hget(name, 'status'), "utf-8")
        if job_status in ["stop", "finish"]:
            return

        # stop job's process
        job_pid = int(self._redis_connection.hget(name, 'pid'))
        try:
            job_process = psutil.Process(job_pid)
            job_process.suspend()
            self._redis_connection.hset(name, 'status', 'stop')
        except Exception as e:
            raise KeyError(f"Failure to stop the job {name}, due to {e}.")

    def restart_job(self, name: str):
        # Check job's list contain given job
        self._name_validation(name)

        # check job status
        job_status = str(self._redis_connection.hget(name, 'status'), "utf-8")
        assert job_status == "stop", "Restart only for the stop jobs."

        # restart job's process
        job_pid = int(self._redis_connection.hget(name, 'pid'))
        try:
            job_process = psutil.Process(job_pid)
            job_process.resume()
            self._redis_connection.hset(name, 'status', 'stop')
        except Exception as e:
            raise KeyError(f"Failure to restart the job {name}, due to {e}.")

    def delete_job(self, name: str):
        # Check job's list contain given job
        self._name_validation(name)

        # check job status
        job_status = str(self._redis_connection.hget(name, 'status'), "utf-8")
        if job_status in ["stop", "running"]:
            # kill job's process
            job_pid = int(self._redis_connection.hget(name, 'pid'))
            try:
                job_process = psutil.Process(job_pid)
                job_process.kill()
            except Exception as e:
                raise KeyError(f"Failure to delete the job {name}, due to {e}.")

        # Delete from Redis DB
        self._redis_connection.delete(name)

    def list_job(self):
        all_jobs = self._redis_connection.keys()
        for job_name in all_jobs:
            job_name = str(job_name, "utf-8")
            self._update_job_status(job_name)
            print(f"{job_name}, {self._redis_connection.hgetall(job_name)}")
