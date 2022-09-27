from datetime import datetime
from builtins import KeyError

from .encrypt_request import EncryptedRequest
from .encrypt_manager import EncryptManager
from .utils import get_config, update_config


# TODO: user/passwd store and validation
# TODO: file copy between local and remote


default_hostname = "0.0.0.0"
default_api_server_port = "0000"


class QuICTRemoteManager:
    def __init__(
        self,
        hostname: str = default_hostname,
        api_server_port: str = default_api_server_port
    ):
        self._url_prefix = f"http://{hostname}:{api_server_port}/quict"
        self._encrypt = EncryptManager()
        self._encryptedrequest = EncryptedRequest()

    def _validation_login_status(self):
        local_status = get_config()
        if not local_status['login_status']:
            raise KeyError("Please login first.")

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        login_time = local_status['last_login_date']
        time_diff = datetime.strptime(login_time, '%Y-%m-%d %H:%M:%S') -\
            datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S')
        if time_diff.seconds > 3600:
            raise ValueError("Please login again. The last login is expired.")

    def _update_login_status(self, username, login: bool = True):
        local_status = {
            'username': username,
            'login_status': login,
            'last_login_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        update_config(local_status)

    ####################################################################
    ############                Login & Logout              ############
    ####################################################################
    def login(self, username: str, password: str):
        self._update_login_status(username, login=False)
        success = self._encryptedrequest.post(
            f"{self._url_prefix}/env/login",
            {'username': username, 'password': self._encrypt.encrypted_passwd(password)}
        )

        if not success:
            raise ValueError("unmatched login username and password.")
        else:
            self._update_login_status(username)

    def logout(self):
        self._update_login_status(None, False)

    ####################################################################
    ############             Cluster API Function           ############
    ####################################################################
    def status_cluster(self):
        self._validation_login_status()

        url = f"{self._url_prefix}/env/status"
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
