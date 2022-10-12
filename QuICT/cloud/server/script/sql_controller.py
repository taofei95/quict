import sqlite3


class SQLManger:
    """ Using SQL database to store user information and password. """
    def __init__(self):
        self._connect = sqlite3.connect("user_info.db")
        self._cursor = self._connect.cursor()

    def __del__(self):
        self._cursor.close()
        self._connect.close()

    def _create_table(self):
        # Temp add here, delete after create datebase
        self._cursor.execute('CREATE TABLE USER_PASS_MAPPING(NAME CHAR(256) NOT NULL, PASSWORD CHAR(256))')
        self._cursor.execute(
            'CREATE TABLE USER_STATIC_INFO(NAME CHAR(256) NOT NULL, LEVEL INT, MAX_RUNNING_JOB INT, MAX_STOPPED_JOB INT, GPU_ALLOWENCE BOOL)'
        )

    def validation_password(self, username: str, passwd: str):
        # Get Stored User Passwd
        user_passwd = self._cursor.execute(f'SELECT PASSWORD FROM USER_PASS_MAPPING WHERE NAME={username}')

        return user_passwd == passwd

    def add_user(self, username: str, passwd: str, user_info: dict):
        # Add user-passwd into USER_PASS_MAPPING
        self._cursor.execute(f"INSERT INTO USER_PASS_MAPPING VALUES({username}, {passwd})")

        # Add user info into database
        level = user_info['level']
        max_rjobs = user_info['max_running_job']
        max_sjobs = user_info['max_stopped_job']
        gpu_allow = user_info['gpu_allowence']
        self._cursor.execute(f"INSERT INTO USER_STATIC_INFO VALUES({username}, {level}, {max_rjobs}, {max_sjobs}, {gpu_allow})")

        self._connect.commit()

    def get_user_info(self, username: str):
        user_info = self._cursor.execute(f"SELECT * FROM USER_STATIC_INFO WHERE NAME={username}")

        return user_info

    def delete_user(self, username: str):
        try:
            # DELETE FROM USER_PASS_MAPPING
            self._cursor.execute(f"DELETE FROM USER_PASS_MAPPING WHERE NAME={username}")
            # DELETE FROM USER_STATIC_INFO
            self._cursor.execute(f"DELETE FROM USER_STATIC_INFO WHERE NAME={username}")

            self._connect.commit()
        except:
            self._connect.rollback()    # RollBack if failure to delete
