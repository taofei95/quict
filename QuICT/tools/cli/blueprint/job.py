import os
import shutil

from QuICT.tools.cli.utils import path_check
from QuICT.tools.cli.client import QuICTLocalJobManager


# Local Controller
local_job_manager = QuICTLocalJobManager()


@path_check
def get_template(output_path: str):
    tempfile_path = os.path.join(
        os.path.dirname(__file__),
        "../template/quict_job.yml"
    )

    shutil.copy(tempfile_path, output_path)


def start_job(file: str):
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
