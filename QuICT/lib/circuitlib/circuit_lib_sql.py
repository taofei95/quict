import os
import sqlite3


class CircuitLibDB:
    def __init__(self):
        self._file_path = os.path.dirname(__file__)
        self._connect = sqlite3.connect(f"{self._file_path}/user_info.db")
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
        self._cursor.execute("DROP TABLE CIRCUIT_LIB")

    def _create_table(self):
        # Temp add here, delete after create datebase
        self._cursor.execute(
            'CREATE TABLE CIRCUIT_LIB(' +
            'ID INTEGER PRIMARY KEY AUTOINCREMENT, NAME text, TYPE text, CLASSIFY text, ' +
            'WIDTH INT, SIZE INT, DEPTH INT)'
        )

    def add_template_circuit(self):
        file_path = os.path.join(
            self._file_path,
            "circuit_qasm",
            "template"
        )
        for file in filter(lambda x: x.endswith('.qasm'), os.listdir(file_path)):
            _, width, size, depth, _ = file.split("_")
            width = int(width[1:])
            size = int(size[1:])
            depth = int(depth[1:])

            self._cursor.execute(
                "INSERT INTO CIRCUIT_LIB(NAME, TYPE, CLASSIFY, WIDTH, SIZE, DEPTH)" +
                f"VALUES (\'{file}\', \'template\', \'template\', " +
                f"\'{width}\', \'{size}\', \'{depth}\')"
            )

        self._connect.commit()

        print(self._cursor.lastrowid)

    def add_random_circuit(self):
        file_path = os.path.join(
            self._file_path,
            "circuit_qasm",
            "random"
        )

        for classify in os.listdir(file_path):
            if classify == ".keep":
                continue

            folder = os.path.join(file_path, classify)
            for file in filter(lambda x: x.endswith('.qasm'), os.listdir(folder)):
                width, size, depth = file.split("_")
                width = int(width[1:])
                size = int(size[1:])
                depth = int(depth[1:-5])

                self._cursor.execute(
                    "INSERT INTO CIRCUIT_LIB(NAME, TYPE, CLASSIFY, WIDTH, SIZE, DEPTH)" +
                    f"VALUES (\'{file}\', \'random\', \'{classify}\', " +
                    f"\'{width}\', \'{size}\', \'{depth}\')"
                )

        self._connect.commit()

        print(self._cursor.lastrowid)


clsql = CircuitLibDB()
# clsql.clean()
clsql.add_template_circuit()
clsql.add_random_circuit()

