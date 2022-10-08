from QuICT.cloud.client.remote import QuICTRemoteManager
from .helper_function import yaml_decompostion


remote_manager = QuICTRemoteManager()


def login(name: str, password: str):
    remote_manager.login(name, password)


def logout():
    remote_manager.logout()


def status_cluster():
    return remote_manager.status_cluster()


@yaml_decompostion
def start_job(file: dict):
    data = remote_manager.start_job(file)
    print(data)


def stop_job(name: str):
    remote_manager.stop_job(name)


def restart_job(name: str):
    remote_manager.restart_job(name)


def delete_job(name: str):
    data = remote_manager.delete_job(name)
    print(data)

def status_job(name: str):
    remote_manager.status_job(name)


def list_jobs():
    data = remote_manager.list_jobs()
    print(data)
