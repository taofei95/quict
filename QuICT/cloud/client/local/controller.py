import os
import psutil
from collections import defaultdict, namedtuple
import multiprocessing as mp

from QuICT.core import Circuit
from QuICT.simulation import Simulator
from QuICT.tools.interface import OPENQASMInterface


JOBINFO = namedtuple('JOBINFO', 'pid type status output_path')


def simulation_start(circuit: Circuit, simualtor_options: dict, output_path: str):
    pass


def qcda_start(circuit: Circuit, qcda_options: dict, output_path: str):
    pass


class QuICTLocalJobManager:
    def __init__(self):
        self._job_queue = defaultdict(dict)

    def start_job(self, yml_dict: dict):
        # Check job name
        name = yml_dict["name"]
        if name in self._job_queue.keys():
            raise KeyError("Repeated name in local jobs.")

        # Get circuit
        cir_path = yml_dict['circuit']
        try:
            circuit = OPENQASMInterface.load_file(cir_path).circuit
        except:
            raise KeyError(f"Failure to load circuit from {cir_path}.")

        # Create Job in job_queue
        job_type = yml_dict['type']
        job_info = JOBINFO(0, job_type, 'initialing', yml_dict['output_path'])
        self._job_queue[name] = job_info

        # Start job by its purpose
        self._start_job(
            name, circuit, yml_dict["simulation"], yml_dict['output_path']
        )

    def _start_job(
        self,
        name: str,
        circuit: Circuit,
        options: dict,
        output_path: str
    ):
        job_info: JOBINFO = self._job_queue[name]
        circuit = job_info.circuit

        # Start process for current job
        target_function = simulation_start if job_info.type == "simulation" else qcda_start
        process = mp.Process(target=target_function, args=(circuit, options, output_path))
        process.start()
        child_pid = process.pid

        # Update job status
        job_info.pid = child_pid
        job_info.status = 'running'

    def status_job(self, name: str):
        # Check job name
        if name not in self._job_queue.keys():
            raise KeyError(
                f"Unmatched job name {name} in local jobs, please using \'quict local job list\' first."
            )

        self._update_job_status(name)

        return self._job_queue[name]

    def stop_job(self, name: str):
        # Check job name
        if name not in self._job_queue.keys():
            raise KeyError(
                f"Unmatched job name {name} in local jobs, please using \'quict local job list\' first."
            )

        # check job status
        job_info: JOBINFO = self._job_queue[name]
        if job_info.status in ["stop", "finish"]:
            return

        # stop job's process
        job_pid = job_info.pid
        try:
            job_process = psutil.Process(job_pid)
            job_process.suspend()
            job_info.status = "stop"
        except Exception as e:
            raise KeyError(f"Failure to stop the job {name}, due to {e}.")

    def restart_job(self, name: str):
        # Check job name
        if name not in self._job_queue.keys():
            raise KeyError(
                f"Unmatched job name {name} in local jobs, please using \'quict local job list\' first."
            )

        # check job status
        job_info: JOBINFO = self._job_queue[name]
        assert job_info.status == "stop", "Restart only for the stop jobs."

        # restart job's process
        job_pid = job_info.pid
        try:
            job_process = psutil.Process(job_pid)
            job_process.resume()
            job_info.status = "running"
        except Exception as e:
            raise KeyError(f"Failure to restart the job {name}, due to {e}.")

    def list_job(self):
        for name, job_info in self._job_queue.values():
            if job_info.status != "finish":
                self._update_job_status(name)

        return self._job_queue

    def _update_job_status(self, name: str):
        job_info: JOBINFO = self._job_queue[name]
        job_pid = job_info.pid
        try:
            job_process = psutil.Process(job_pid)
            job_info.status = job_process.status()
        except:
            job_info.status = "finish"
