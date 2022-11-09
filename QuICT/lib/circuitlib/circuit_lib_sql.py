import os
import sqlite3


class CircuitLibDB:
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
        """ Clean the table data in current DB. """
        self._cursor.execute("DROP TABLE CIRCUIT_LAB")

    def _create_table(self):
        # Temp add here, delete after create datebase
        self._cursor.execute(
            'CREATE TABLE CIRCUIT_LAB(' +
            'ID INTEGER PRIMARY KEY AUTOINCREMENT, QASMNAME text, TYPE text unique, CLASSIFY text unique, ' +
            'WIDTH INT, SIZE INT, DEPTH INT)'
        )

    def add_circuit(self, file_path):
        pass
