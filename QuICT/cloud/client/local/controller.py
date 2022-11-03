import os
import psutil
import subprocess

from QuICT.tools import Logger
from QuICT.tools.logger import LogFormat
from .sql_manage_local import SQLMangerLocalMode


logger = Logger("Job_Management_Local_Mode", LogFormat.full)


class QuICTLocalJobManager:
    """ QuICT Job Management for the Local Mode. Using SQL to store running-time information. """
    def __init__(self):
        self._sql_connect = SQLMangerLocalMode()

    def _name_validation(self, name) -> bool:
        if not self._sql_connect.job_validation(name):
            logger.warn(
                f"Unmatched job name {name} in local jobs, please using \'quict local job list\' first."
            )
            return False

        # Update given job's status
        self._update_job_status(name)
        return True

    def start_job(self, yml_dict: dict):
        """ Start the job describe by the given yaml file.

        Args:
            yml_dict (dict): The given yaml file

        Raises:
            KeyError: Key Error in given yaml file
        """
        # Check job name
        name = yml_dict["job_name"]
        if self._sql_connect.job_validation(name):
            logger.warn("Repeated name in local jobs.")
            return

        try:
            self._start_job(name, yml_dict)
        except Exception as e:
            logger.warn(f"Failure to start job, due to {e}.")

    def _start_job(self, job_name: str, yml_dict: dict):
        # Get information from given yml dict
        cir_path = yml_dict['circuit']
        job_type = yml_dict['type']
        job_options = yml_dict["simulation"] if job_type == "simulation" else yml_dict["qcda"]
        device = yml_dict["resource"]["device"]
        output_path = yml_dict['output_path']

        # Pre-paration job's runtime arguments
        if job_type == "simulation":
            shots = job_options["shots"]
            backend = job_options["backend"]
            precision = job_options["precision"]
            runtime_args = f"{cir_path} {shots} {device} {backend} {precision} {output_path}"
        else:
            runtime_args = f"{cir_path} {output_path}"
            optimization = job_options["optimization"]["enable"]
            runtime_args += f" {optimization}"

            if job_options["mapping"]["enable"]:
                layout_path = job_options["mapping"]["layout_path"]
                runtime_args += f" {layout_path}"
            else:
                runtime_args += " False"

            if job_options["synthesis"]["enable"]:
                iset = job_options["synthesis"]["instruction_set"]
                runtime_args += f" {iset}"

        # Start job
        command_file_path = os.path.join(
            os.path.dirname(__file__),
            "../../cli/script",
            f"{job_type}.py"
        )
        proc = subprocess.Popen(
            f"python {command_file_path} {runtime_args}", shell=True
        )

        # Save job information into SQL DB.
        job_info = {
            'name': job_name,
            'status': 'running',
            'pid': proc.pid
        }
        self._sql_connect.add_job(job_info)

        logger.info(f"Successfully start the job {job_name} in local mode.")

    def _update_job_status(self, name: str):
        """ Update job's running status, do not distinguish error and finish.

        Args:
            name (str): job's name
        """
        job_status = self._sql_connect.get_job_status(name)
        if job_status in ["finish", "stop"]:
            return

        job_pid = self._sql_connect.get_job_pid(name)
        try:
            _ = psutil.Process(job_pid)
        except:
            self._sql_connect.change_job_status(name, 'finish')

    def status_job(self, name: str):
        """ Get job's states """
        # Check job's list contain given job
        if self._name_validation(name):
            logger.info(
                f"The job state of {name} is {self._sql_connect.get_job_status(name)}"
            )

    def stop_job(self, name: str):
        """ Stop a job. """
        # Check job's list contain given job
        if not self._name_validation(name):
            return

        # check job status
        job_status = self._sql_connect.get_job_status(name)
        if job_status in ["stop", "finish"]:
            logger.info(f"The job {name} is already {job_status}, no need stop anymore.")
            return

        # stop job's process
        job_pid = self._sql_connect.get_job_pid(name)
        try:
            job_process = psutil.Process(job_pid)
            job_process.suspend()
            self._sql_connect.change_job_status(name, 'stop')
            logger.info(f"Successfully stop the job {name}.")
        except Exception as e:
            logger.warn(f"Failure to stop the job {name}, due to {e}.")

    def restart_job(self, name: str):
        """ Restart a job. """
        # Check job's list contain given job
        if not self._name_validation(name):
            return

        # check job status
        job_status = self._sql_connect.get_job_status(name)
        if job_status != "stop":
            logger.info("Restart only for the stop jobs.")
            return

        # restart job's process
        job_pid = self._sql_connect.get_job_pid(name)
        try:
            job_process = psutil.Process(job_pid)
            job_process.resume()
            self._sql_connect.change_job_status(name, "running")
            logger.info(f"Successfully restart the job {name}.")
        except Exception as e:
            logger.warn(f"Failure to restart the job {name}, due to {e}.")

    def delete_job(self, name: str):
        """ Delete a job. """
        # Check job's list contain given job
        if not self._name_validation(name):
            return

        # check job status
        job_status = self._sql_connect.get_job_status(name)
        # kill job's process, if job is still running
        if job_status in ["stop", "running"]:
            job_pid = self._sql_connect.get_job_pid(name)
            try:
                job_process = psutil.Process(job_pid)
                job_process.kill()
                logger.info(f"Successfully kill the job {name}'s process.")
            except Exception as e:
                logger.warn(f"Failure to kill the job {name}, due to {e}.")

        # Delete from Redis DB
        self._sql_connect.delete_job(name)
        logger.info(f"Successfully delete the job {name}.")

    def list_job(self):
        """ List all jobs. """
        all_jobs = self._sql_connect.list_jobs()
        logger.info(f"There are {len(all_jobs)} jobs in local mode.")
        for job_name, _, pid in all_jobs:
            self._update_job_status(job_name)
            logger.info(
                f"job name: {job_name}, related pid: {pid}, " +
                f"job's state: {self._sql_connect.get_job_status(job_name)}"
            )
