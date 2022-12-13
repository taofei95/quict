import os
import subprocess


command_file_path = os.path.join(
    os.path.dirname(__file__),
    "../script/benchmark.py"
)


def algorithm():
    _ = subprocess.call(
        f"python {command_file_path} algorithm", shell=True
    )


def qcda(circuit_path):
    _ = subprocess.call(
        f"python {command_file_path} qcda {circuit_path}", shell=True
    )


def simulation(circuit_path, gpu):
    _ = subprocess.call(
        f"python {command_file_path} simulation {circuit_path} {gpu}", shell=True
    )
