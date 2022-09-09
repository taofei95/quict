import os
import shutil

from QuICT.cloud.client.local import QuICTLocalJobManager
from .decorator import path_check, yaml_decompostion

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


# TODO: add decomposition for given job yaml file
@yaml_decompostion
def start_job(file: dict):
    mode = file['mode']
    if mode == "local":
        return local_job_manager.start_job(file)
    elif mode == "remote":
        pass
    else:
        raise KeyError(f"unrecognized mode {mode} in CLI, please use [local, remote].")


def stop_job(name: str):
    pass


def restart_job(name: str):
    pass


def delete_job(name: str):
    pass


def status_job(name: str):
    pass


def list_jobs():
    pass
