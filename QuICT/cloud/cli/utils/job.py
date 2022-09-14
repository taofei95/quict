import os
import shutil

from QuICT.cloud.client.local import QuICTLocalJobManager
from .helper_function import path_check, yaml_decompostion

# Local Controller
local_job_manager = QuICTLocalJobManager()
# Remote Controller


@path_check
def get_template(type: str, output_path: str):
    tempfile_path = os.path.join(
        os.path.dirname(__file__),
        "../template"
    )
    type_list = ["simulation", "qcda"] if type == "all" else [type]
    for t in type_list:
        file_name = f"job_{t}.yml"
        temp_file_path = os.path.join(
            tempfile_path,
            file_name
        )
        shutil.copy(temp_file_path, output_path)


@yaml_decompostion
def start_job(file: dict, mode: str):
    if mode == "local":
        return local_job_manager.start_job(file)
    elif mode == "remote":
        pass
    else:
        raise KeyError(f"unrecognized mode {mode} in CLI, please use [local, remote].")


def stop_job(name: str, mode: str):
    if mode == "local":
        return local_job_manager.stop_job(name)
    elif mode == "remote":
        pass
    else:
        raise KeyError(f"unrecognized mode {mode} in CLI, please use [local, remote].")


def restart_job(name: str, mode: str):
    if mode == "local":
        return local_job_manager.restart_job(name)
    elif mode == "remote":
        pass
    else:
        raise KeyError(f"unrecognized mode {mode} in CLI, please use [local, remote].")


def delete_job(name: str, mode: str):
    if mode == "local":
        return local_job_manager.delete_job(name)
    elif mode == "remote":
        pass
    else:
        raise KeyError(f"unrecognized mode {mode} in CLI, please use [local, remote].")


def status_job(name: str, mode: str):
    if mode == "local":
        return local_job_manager.status_job(name)
    elif mode == "remote":
        pass
    else:
        raise KeyError(f"unrecognized mode {mode} in CLI, please use [local, remote].")


def list_jobs(mode: str):
    if mode == "local":
        return local_job_manager.list_job()
    elif mode == "remote":
        pass
    else:
        raise KeyError(f"unrecognized mode {mode} in CLI, please use [local, remote].")
