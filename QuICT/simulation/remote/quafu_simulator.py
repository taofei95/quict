# ----------------------------
# Quafu
# http://quafu.baqis.ac.cn/
# ----------------------------
from quafu import User, Task, QuantumCircuit


class QuafuSimulator:
    __BACKEND = ["ScQ-P10", "ScQ-P18", "ScQ-P136"]
    def __init__(self, token: str):
        self.user = User()
        self.user.save_apitoken(token)

    def run(self, circuit, backend: str = "ScQ-P10", shots: int = 1000, compile: bool = True):
        qc = QuantumCircuit(circuit.width())
        test_cir = circuit.qasm()
        qc.from_openqasm(test_cir)

        assert backend in self.__BACKEND
        task = Task()
        task.config(backend=backend, shots=shots, compile=compile)
        res = task.send(qc)

        return res.counts
