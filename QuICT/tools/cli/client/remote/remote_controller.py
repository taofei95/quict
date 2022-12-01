from QuICT.cloud.client.local.sql_manage_local import SQLMangerLocalMode
from QuICT.tools import Logger
from QuICT.tools.logger import LogFormat
from .encrypt_request import EncryptedRequest
from .encrypt_manager import EncryptManager


# TODO: file copy between local and remote
default_hostname = "127.0.0.1"
default_api_server_port = "5000"


def login_validation(function):
    def decorator(self, *args, **kwargs):
        login, err_msg = self._sql_db.login_validation()
        if not login:
            self._logger.warn(err_msg)
            return

        login_info = self._sql_db.get_login_info()

        result = function(self, user_info=login_info, *args, **kwargs)

        return result

    return decorator


class QuICTRemoteManager:
    """ QuICT Job Management for the Remote Mode. """
    def __init__(
        self,
        hostname: str = default_hostname,
        api_server_port: str = default_api_server_port
    ):
        self._url_prefix = f"http://{hostname}:{api_server_port}/quict"
        self._encrypt = EncryptManager()
        self._encryptedrequest = EncryptedRequest()
        self._sql_db = SQLMangerLocalMode()
        self._logger = Logger("Job_Management_Remote_Mode", LogFormat.full)

    ####################################################################
    ############                Login & Logout              ############
    ####################################################################
    def login(self, username: str, password: str):
        """ User login.

        Args:
            username (str): The username
            password (str): The password
        """
        encrypted_passwd = self._encrypt.encrypted_passwd(password)
        success = self._encryptedrequest.post(
            f"{self._url_prefix}/env/login",
            {'username': username, 'password': encrypted_passwd},
            (1, username, encrypted_passwd),
            is_login=True
        )

        if not success:
            self._logger.warn("unmatched login username and password.")
        else:
            self._sql_db.user_login(username, encrypted_passwd)

    def logout(self):
        """ User logout. """
        self._sql_db.user_logout()

    def register(
        self, username: str, password: str, email: str, level: int = 3
    ):
        """ User register.

        Args:
            username (str): The username
            password (str): The password
            email (str): The email address
            level (int): The difficult level
        """
        encrypted_passwd = self._encrypt.encrypted_passwd(password)
        success = self._encryptedrequest.post(
            f"{self._url_prefix}/env/register",
            {
                'username': username,
                'password': encrypted_passwd,
                'email': email,
                'level': level
            },
            (1, username, encrypted_passwd),
            is_login=True
        )

        if not success:
            self._logger.warn("Failure to register with current username.")
        else:
            self._sql_db.user_login(username, encrypted_passwd)

    def unsubscribe(self, username: str, password: str):
        """ delete user's account. """
        encrypted_passwd = self._encrypt.encrypted_passwd(password)
        success = self._encryptedrequest.post(
            f"{self._url_prefix}/env/unsubscribe",
            {
                'username': username,
                'password': encrypted_passwd,
            },
            (1, username, encrypted_passwd),
            is_login=False
        )

        if not success:
            self._logger.warn("Failure to unsubscribe with current username.")

    ####################################################################
    ############               Job API Function             ############
    ####################################################################
    @login_validation
    def start_job(self, yml_dict: dict, user_info: tuple):
        """ Send job ticket to cloud system.

        Args:
            yml_dict (dict): The job's information
            user_info (dict): The user's information
        """
        # Delete Circuit Qasm Path here, not use for remote mode
        self._remote_job_validate(yml_dict)

        url = f"{self._url_prefix}/jobs/start"
        try:
            _ = self._encryptedrequest.post(url, yml_dict, user_info=user_info)
            self._logger.info("Successfully send job to cloud.")
        except Exception as e:
            self._logger.warn(f"Failure to start target job, due to {e}.")

    def _remote_job_validate(self, yml_dict: dict):
        yml_dict['device'] = 'CPU' if yml_dict["type"] == "qcda" else yml_dict["simulation"]["device"]
        # Deal with circuit, load circuit's qasm.
        with open(yml_dict['circuit']) as cfile:
            circuit_data = cfile.read()

        del yml_dict['circuit']
        yml_dict['circuit_info']['qasm'] = circuit_data

        # Deal with layout
        if yml_dict['type'] == "qcda" and "layout_path" in yml_dict["qcda"].keys():
            with open(yml_dict['qcda']['layout_path']) as lfile:
                layout_data = lfile.read()

            del yml_dict['qcda']['layout_path']
            yml_dict['qcda']['layout_string'] = layout_data

    @login_validation
    def status_job(self, job_name: str, user_info: tuple):
        """ Check a job's running states

        Args:
            job_name (str): The job's name
            user_info (tuple): The user's information
        """
        url = f"{self._url_prefix}/jobs/{job_name}:status"
        state = self._encryptedrequest.get(url, user_info=user_info)
        self._logger.info(state)

    @login_validation
    def stop_job(self, job_name: str, user_info: tuple):
        """ Stop a running job.

        Args:
            job_name (str): The job's name
            user_info (tuple): The user's information
        """
        url = f"{self._url_prefix}/jobs/{job_name}:stop"
        try:
            self._encryptedrequest.post(url, user_info=user_info)
            self._logger.info("Successfully send stop request to cloud.")
        except Exception as e:
            self._logger.warn(f"Failure to stop target job, due to {e}.")

    @login_validation
    def restart_job(self, job_name: str, user_info: tuple):
        """ Restart a running job.

        Args:
            job_name (str): The job's name
            user_info (tuple): The user's information
        """
        url = f"{self._url_prefix}/jobs/{job_name}:restart"
        try:
            self._encryptedrequest.post(url, user_info=user_info)
            self._logger.info("Successfully send restart request to cloud.")
        except Exception as e:
            self._logger.warn(f"Failure to restart target job, due to {e}.")

    @login_validation
    def delete_job(self, job_name: str, user_info: tuple):
        """ Delete a job.

        Args:
            job_name (str): The job's name
            user_info (tuple): The user's information
        """
        url = f"{self._url_prefix}/jobs/{job_name}:delete"
        try:
            self._encryptedrequest.delete(url, user_info=user_info)
            self._logger.info("Successfully send delete request to cloud.")
        except Exception as e:
            self._logger.warn(f"Failure to delete target job, due to {e}.")

    @login_validation
    def list_jobs(self, user_info: tuple):
        """ List all jobs in the cloud system.

        Args:
            user_info (tuple): The user's information
        """
        url = f"{self._url_prefix}/jobs/list"
        self._logger.info(self._encryptedrequest.get(url, user_info=user_info))
