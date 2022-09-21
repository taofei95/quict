from QuICT.cloud.client.remote import QuICTRemoteManager
from .helper_function import yaml_decompostion


def login(username: str, password: str):
    global remote_manager
    remote_manager = QuICTRemoteManager(username, password)
    remote_manager.login()


def logout():
    remote_manager.logout()


def status_cluster():
    return remote_manager.status_cluster()


@yaml_decompostion
def start_job(file: str):
    remote_manager.start_job(file)


def stop_job(name: str):
    remote_manager.stop_job(name)


def restart_job(name: str):
    remote_manager.restart_job(name)


def delete_job(name: str):
    remote_manager.delete_job(name)


def status_job(name: str):
    remote_manager.status_job(name)


def list_jobs():
    remote_manager.list_jobs()
