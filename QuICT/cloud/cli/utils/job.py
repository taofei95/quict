import os
import shutil

from QuICT.cloud.client.local import QuICTLocalJobManager
from .helper_function import path_check, yaml_decompostion

# Local Controller
local_job_manager = QuICTLocalJobManager()


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
def start_job(file: dict):
    local_job_manager.start_job(file)


def stop_job(name: str):
    local_job_manager.stop_job(name)


def restart_job(name: str):
    local_job_manager.restart_job(name)


def delete_job(name: str):
    local_job_manager.delete_job(name)


def status_job(name: str):
    local_job_manager.status_job(name)


def list_jobs():
    local_job_manager.list_job()
