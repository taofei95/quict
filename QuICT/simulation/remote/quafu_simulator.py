# ----------------------------
# Quafu
# http://quafu.baqis.ac.cn/
# ----------------------------
from quafu import User, Task, QuantumCircuit


class QuafuSimulator:
    """ Quafu Quantum Machine Interface. """
    __BACKEND = ["ScQ-P10", "ScQ-P18", "ScQ-P136"]

    def __init__(self, token: str):
        """ Initial QuafuSimulator Class.

        Args:
            token (str): Personal Token for Quafu Platform Login.
        """
        self.user = User()
        self.user.save_apitoken(token)

    def run(self, circuit, backend: str = "ScQ-P10", shots: int = 1000, compile: bool = True):
        """ start quafu quantum machine with given circuit

        Args:
            circuit (Circuit): The quantum circuits.
            backend (str, optional): The backend choice. Defaults to "ScQ-P10".
            shots (int, optional): The sample times. Defaults to 1000.
            compile (bool, optional): Whether use Quafu's compiler or not. Defaults to True.

        Returns:
            list: The sample result
        """
        qc = QuantumCircuit(circuit.width())
        test_cir = circuit.qasm()
        qc.from_openqasm(test_cir)

        assert backend in self.__BACKEND
        task = Task()
        task.config(backend=backend, shots=shots, compile=compile)
        res = task.send(qc)

        return res.counts
