import redis

from .encrypt_request import EncryptedRequest


# TODO: user/passwd store and validation
# TODO: file copy between local and remote

default_hostname = "0.0.0.0"
default_api_server_port = "0000"


class QuICTRemoteManager:
    def __init__(
        self,
        username: str,
        password: str,
        hostname: str = default_hostname,
        api_server_port: str = default_api_server_port
    ):
        self._url_prefix = f"http://{hostname}:{api_server_port}/quict"
        self._username = username
        self._password = password
        self._encryptedrequest = EncryptedRequest(username)

    ####################################################################
    ############                Login & Logout              ############
    ####################################################################
    def login(self):
        pass

    def logout(self):
        pass

    ####################################################################
    ############             Cluster API Function           ############
    ####################################################################
    def status_cluster(self):
        url = f"{self._url_prefix}/cluster/status"
        return self._encryptedrequest.get(url)

    ####################################################################
    ############               Job API Function             ############
    ####################################################################
    def start_job(self, yml_dict: dict):
        url = f"{self._url_prefix}/job/start"

        # TODO: related file copy to server

        return self._encryptedrequest.post(url, yml_dict)

    def status_job(self, job_name: str):
        url = f"{self._url_prefix}/job/{job_name}:status"
        return self._encryptedrequest.get(url)

    def stop_job(self, job_name: str):
        url = f"{self._url_prefix}/job/{job_name}:stop"
        return self._encryptedrequest.post(url)

    def restart_job(self, job_name: str):
        url = f"{self._url_prefix}/job/{job_name}:restart"
        return self._encryptedrequest.post(url)
    
    def delete_job(self, job_name: str):
        url = f"{self._url_prefix}/job/{job_name}:delete"
        return self._encryptedrequest.delete(url)

    def list_jobs(self):
        url = f"{self._url_prefix}/job/list"
        return self._encryptedrequest.get(url)
