import requests


hostname = "0.0.0.0"
port = 8888
job_prefix = f"http://{hostname}:{port}/v1/jobs"


def start_job(file: str):
    # requests.post(
    #     url=f"{job_prefix}/start",
    #     data=file
    # )
    print(file)


def stop_job(name: str):
    requests.post(
        url=f"{job_prefix}/{name}:stop"
    )


def restart_job(name: str):
    requests.post(
        url=f"{job_prefix}/{name}:restart"
    )


def delete_job(name: str):
    requests.post(
        url=f"{job_prefix}/{name}:delete"
    )


def status_job(name: str):
    requests.get(
        url=f"{job_prefix}/{name}:status"
    )


def list_jobs():
    requests.get(
        url=f"{job_prefix}/list"
    )
