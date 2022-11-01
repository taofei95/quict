import os
import sqlite3


def sql_locked(function):
    def decorator(self, *args, **kwargs):
        self._connect.execute('BEGIN EXCLUSIVE')
        result = function(self, *args, **kwargs)
        self._connect.commit()

        return result

    return decorator


class SQLMangerLocalMode:
    """ Using SQL database to store running job information in local mode, and
    login information for remote mode.
    """
    def __init__(self):
        file_path = os.path.dirname(__file__)
        self._connect = sqlite3.connect(f"{file_path}/user_info.db")
        self._connect.isolation_level = "EXCLUSIVE"
        self._cursor = self._connect.cursor()

        # Initial Database when first run
        try:
            self._create_table()
        except:
            pass

    def __del__(self):
        self._cursor.close()
        self._connect.close()

    def clean(self):
        self._cursor.execute("DROP TABLE JOB_INFO")
        self._cursor.execute("DROP TABLE USER_LOGIN_INFO")

    def _create_table(self):
        # Temp add here, delete after create datebase
        self._cursor.execute('CREATE TABLE JOB_INFO(NAME text unique, STATUS text, PID INT)')
        self._cursor.execute(
            'CREATE TABLE USER_LOGIN_INFO(NAME text unique, PASSWORD text, LOGINTIME text, STATUS bool)'
        )

    ####################################################################
    ############               User DB Function             ############
    ####################################################################

    def get_password(self, username: str) -> str:
        self._cursor.execute(f'SELECT PASSWORD FROM USER_LOGIN_INFO WHERE NAME=\'{username}\'')
        user_passwd = self._cursor.fetchall()[0]

        return user_passwd[0]

    def validate_user(self, username: str) -> bool:
        user_data = self.get_user_info(username)

        return user_data is not None

    @sql_locked
    def update_password(self, username: str, new_passwd: str):
        self._cursor.execute(f"UPDATE USER_PASS_MAPPING SET PASSWORD = \'{new_passwd}\' WHERE NAME = \'{username}\'")

    def get_user_info(self, username: str) -> tuple:
        self._cursor.execute(f"SELECT * FROM USER_STATIC_INFO WHERE NAME=\'{username}\'")
        user_info = self._cursor.fetchone()

        return user_info

    ####################################################################
    ############               Job DB Function              ############
    ####################################################################

    def job_validation(self, job_name: str) -> bool:
        job_info = self._cursor.execute(f"SELECT * FROM JOB_INFO WHERE NAME=\'{job_name}\'")
        job_info = self._cursor.fetchone()

        return job_info is not None

    @sql_locked
    def add_job(self, job_info: dict):
        name = job_info['name']     # Job name
        status = job_info['status']
        pid = job_info['pid']

        # Add job information into JOB_INFO
        self._cursor.execute(f"INSERT INTO JOB_INFO VALUES (\'{name}\', \'{status}\', \'{pid}\')")

    @sql_locked
    def get_job_status(self, job_name: str):
        self._cursor.execute(f'SELECT STATUS FROM JOB_INFO WHERE NAME=\'{job_name}\'')
        job_status = self._cursor.fetchone()

        return job_status[0]

    @sql_locked
    def get_job_pid(self, job_name: str):
        self._cursor.execute(f'SELECT PID FROM JOB_INFO WHERE NAME=\'{job_name}\'')
        job_status = self._cursor.fetchone()

        return job_status[0]

    @sql_locked
    def change_job_status(self, job_name: str, job_status: str):
        self._cursor.execute(f"UPDATE JOB_INFO SET STATUS = \'{job_status}\' WHERE NAME = \'{job_name}\'")

    @sql_locked
    def delete_job(self, job_name: str):
        try:
            # DELETE FROM USER_PASS_MAPPING
            self._cursor.execute(f"DELETE FROM JOB_INFO WHERE NAME=\'{job_name}\'")
        except:
            self._connect.rollback()    # RollBack if failure to delete

    def list_jobs(self):
        self._cursor.execute('SELECT * FROM JOB_INFO')
        return self._cursor.fetchall()
