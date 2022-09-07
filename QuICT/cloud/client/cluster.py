import requests


hostname = "0.0.0.0"
port = 8888
job_prefix = f"http://{hostname}:{port}/v1/cluster"


def status_cluster():
    requests.get(
        url=f"{job_prefix}/status"
    )
