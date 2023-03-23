import os
import time

from QuICT.tools.cli import JobCreator
from QuICT.tools.cli.client import QuICTLocalManager


def create_job_through_JobCreator(save: bool = False):
    path = os.path.join(os.path.dirname(__file__), "cli_example")
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
        layout_path=os.path.join(path, "ibmqx2_layout.json")
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

    # Start Job Through QuICTLocalManager
    job_manager = QuICTLocalManager()
    job_manager.start_job(job_dict)

    # Using Yaml file [a example about how to write yml file for quict cli, need fixed the file path before using]
    # file_path = os.path.join(os.path.dirname(__file__), "cli_example/quict_job.yml")
    # job_manager.start_job(file_path)

    # Check Job's Status
    job_manager.status_job(job_name)

    # List all Job's in local mode
    time.sleep(5)
    job_manager.list_job()

    # Job control
    job_manager.delete_job(job_name)
    job_manager.list_job()


if __name__ == "__main__":
    job_controller_in_local_mode()
