import os
import redis
import psutil
import subprocess


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

        # Get information from given yml dict
        cir_path = yml_dict['circuit']
        job_type = yml_dict['type']
        job_options = yml_dict["simulation"] if job_type == "simulation" else yml_dict["qcda"]
        output_path = yml_dict['output_path']

        # Set job information into Redis.
        job_info = {
            'type': job_type,
            'circuit': cir_path,
            'status': 'initializing',
            'output_path': output_path,
            'pid': -1
        }
        self._redis_connection.hmset(name, job_info)

        # Start job by its purpose
        command_file_path = os.path.join(
            os.path.dirname(__file__),
            "../../cli/script",
            f"{job_type}.py"
        )
        if job_type == "simulation":
            shots = job_options["shots"]
            device = job_options["resource"]["device"]
            backend = job_options["backend"]
            precision = job_options["precision"]
            runtime_args = f"{cir_path} {shots} {device} {backend} {precision} {output_path}"
        else:
            runtime_args = f"{cir_path} {output_path}"
            if job_options["mapping"]["enable"]:
                layout_path = job_options["layout_path"]
                runtime_args += f" {layout_path}"

            if not job_options["optimization"]["enable"]:
                runtime_args += " False"

            if job_options["synthesis"]["enable"]:
                iset = job_options["synthesis"]["instruction_set"]
                runtime_args += f" {iset}"

        _ = subprocess.call(f"python {command_file_path} {name} {runtime_args}", shell=True)

    def _update_job_status(self, name: str):
        job_status = str(self._redis_connection.hget(name, 'status'), "utf-8")
        if job_status in ["finish", "stop"]:
            return

        job_pid = int(self._redis_connection.hget(name, 'pid'))
        try:
            _ = psutil.Process(job_pid)
        except:
            self._redis_connection.hset(name, 'status', 'finish')

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
