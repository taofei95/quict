import os
import psutil
import time
from collections import defaultdict, namedtuple
import multiprocessing as mp

from QuICT.core import Circuit
from QuICT.simulation import Simulator
from QuICT.tools.interface import OPENQASMInterface
from QuICT.cloud.cli.utils.helper_function import name_validation


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
        if name in list(self._job_queue.keys()):
            raise KeyError("Repeated name in local jobs.")

        # Get circuit
        cir_path = yml_dict['circuit']
        circuit = OPENQASMInterface.load_file(cir_path).circuit

        # Create Job in job_queue
        job_type = yml_dict['type']
        job_info = JOBINFO(0, job_type, 'initialing', yml_dict['output_path'])
        self._job_queue[name] = job_info

        # Start job by its purpose
        runtime_args = yml_dict["simulation"] if job_type == "simulation" else yml_dict["qcda"]
        self._start_job(name, circuit, runtime_args)

    def _start_job(
        self,
        name: str,
        circuit: Circuit,
        options: dict
    ):
        job_info: JOBINFO = self._job_queue[name]
        output_path = job_info.output_path

        # Start process for current job
        target_function = simulation_start if job_info.type == "simulation" else qcda_start
        process = mp.Process(target=target_function, args=(circuit, options, output_path))
        process.start()
        child_pid = process.pid

        # Update job status
        job_info.pid = child_pid
        job_info.status = 'running'

    def _update_job_status(self, name: str):
        job_info: JOBINFO = self._job_queue[name]
        job_pid = job_info.pid
        try:
            job_process = psutil.Process(job_pid)
            job_info.status = job_process.status()
        except:
            job_info.status = "finish"

    @name_validation
    def status_job(self, name: str):
        self._update_job_status(name)

        return self._job_queue[name]

    @name_validation
    def stop_job(self, name: str):
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

    @name_validation
    def restart_job(self, name: str):
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

    @name_validation
    def delete_job(self, name: str):
        # check job status
        job_info: JOBINFO = self._job_queue[name]
        if job_info.status in ["stop", "running"]:
            # stop job's process
            job_pid = job_info.pid
            try:
                job_process = psutil.Process(job_pid)
                job_process.kill()
            except Exception as e:
                raise KeyError(f"Failure to delete the job {name}, due to {e}.")

        del self._job_queue[name]

    def list_job(self):
        for name, job_info in self._job_queue.values():
            if job_info.status not in ["finish", "stop"]:
                self._update_job_status(name)

        # Display job queue
        print(self._job_queue)

        return self._job_queue
