import os
import time

from QuICT.tools.cli import JobCreator, shortcut_for_quict
from QuICT.tools.cli.client import QuICTLocalManager, QuICTRemoteManager


def create_job_through_JobCreator(save: bool = False):
    path = os.path.join(os.path.dirname(__file__), "../tempcir")
    path = os.path.abspath(path)
    job_example = JobCreator(
        name="quict-example",
        circuit_path=os.path.join(path, "clifford.qasm"),
        device="CPU"
    )
    job_example.set_simulation_spec(
        shots=1000,
        precision="single",
        backend="state_vector"
    )
    job_example.set_qcda_spec(
        ["Clifford", "GateTransform"],
        instruction_set="USTC",
        auto_mode="light",
        layout_path=os.path.join(path, "../example/layout/ibmqx2_layout.json")
    )

    # Save current job file into yaml file
    if save:
        job_example.to_yaml()

    return job_example.job_dict


def job_controller_in_local_mode():
    # Step 1: Prepare job yaml file
    # Using JobCreator
    job_dict = create_job_through_JobCreator()
    job_name = job_dict["job_name"]
    # Using Yaml file
    file_path = os.path.join(os.path.dirname(__file__), "../tempcir/quict_job.yml")

    # Start Job Through QuICTLocalManager
    job_manager = QuICTLocalManager()
    job_manager.start_job(job_dict)
    job_manager.start_job(file_path)

    # Check Job's Status
    job_manager.status_job(job_name)

    # List all Job's in local mode
    time.sleep(5)
    job_manager.list_job()

    # Job control
    job_manager.delete_job(job_name)
    job_manager.list_job()


def job_controller_in_remote_mode():
    job_dict = create_job_through_JobCreator()
    job_name = job_dict["job_name"]
    # Using Yaml file
    file_path = os.path.join(os.path.dirname(__file__), "../tempcir/quict_job.yml")

    # Start Job Through QuICTLocalManager
    job_manager = QuICTRemoteManager()

    # Register User
    job_manager.register(
        "{your_username}",
        "{your_password}",
        "{your_email}"
    )
    # User Login
    job_manager.login("{your_username}", "{your_password}")

    # Start Job
    time.sleep(2)
    job_manager.start_job(job_dict)
    job_manager.start_job(file_path)

    # Check Job States
    time.sleep(2)
    job_manager.status_job(job_name)

    # List all Job within current User
    job_manager.list_jobs()

    # Delete job
    job_manager.delete_job(job_name)
    time.sleep(15)
    job_manager.list_jobs()


def add_shortcut_quict():
    shortcut_for_quict()


job_controller_in_remote_mode()
