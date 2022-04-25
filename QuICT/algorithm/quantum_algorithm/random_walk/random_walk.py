from QuICT.core import Circuit
from QuICT.core.gate import *


class QuantumRandomWalk:
    _DEFAULT_COIN_OPERATOR = {
        "h": H,
        "i": ID,
        "x": X,
        "y": Y,
        "z": Z
    }

    def __init__(self, T: int, p_space: int, dim: int = 1, coin_operator: str = "h"):
        self.step = T
        self.p_space = p_space
        self.qubits = int(np.ceil(np.log2(p_space)))
        self.dim = dim
        self._coin_operator = self._DEFAULT_COIN_OPERATOR[coin_operator]

        self._circuit_construct()

    def _circuit_construct(self):
        rw_circuit = Circuit(self.qubits + self.dim)
        is_squared = (self.p_space == 2 ** self.qubits)
        for _ in range(self.step):
            self._coin_operator | rw_circuit(self.qubits)
            self.shift_operator() | rw_circuit
            if not is_squared:
                self._incre_mod() | rw_circuit

            self.shift_operator(True) | rw_circuit
            if not is_squared:
                self._decre_mod() | rw_circuit

        self._circuit = rw_circuit

    def _mct_generator(self, qubits, target: int = 0):
        _1 = (1 << qubits) - 1
        _0 = _1 & ((1 << target) - 1) + (_1 >> (target + 1) << (target + 1))

        unitary = np.identity(1 << qubits, dtype=np.complex128)
        unitary[_1, _1], unitary[_0, _0] = np.complex128(0), np.complex128(0)
        unitary[_0, _1], unitary[_1, _0] = np.complex128(1), np.complex128(1)

        return unitary

    def shift_operator(self, decrement: bool = False):
        shift_cgate = CompositeGate()
        qubits = self.qubits + self.dim

        # Construct shift operator composite gate
        if decrement:
            for i in range(1, qubits, 1):
                X | shift_cgate(i)

        for q_size in range(qubits, 1, -1):
            if q_size > 3:
                ugate = Unitary(self._mct_generator(q_size))
                ugate | shift_cgate(list(range(qubits - 1, qubits - 1 - q_size, -1)))

            if q_size == 3:
                CCX | shift_cgate(list(range(qubits - 1, qubits - 4, -1)))

            if q_size == 2:
                CX | shift_cgate([qubits - 1, qubits - 2])

            if decrement:
                X | shift_cgate(qubits - q_size + 1)

        return shift_cgate

    def _incre_mod(self):
        p_state = [i for i in range(self.qubits - 1, -1, -1) if self.p_space & (1 << i)]
        _0_state = set(range(self.qubits)) ^ set(p_state)
        
        # Construct increment mod cgate
        inc_mod = CompositeGate()
        for _0 in _0_state:
            X | inc_mod(self.qubits - 1 - _0)

        for idx, pbit in enumerate(p_state):
            if idx != 0:
                X | inc_mod(self.qubits - 1 - p_state[idx - 1])
        
            ugate = Unitary(self._mct_generator(
                self.qubits + self.dim,
                self.qubits - 1 - pbit
            ))
            ugate | inc_mod(list(range(self.qubits + self.dim - 1, -1, -1)))

        for idx, pbit in enumerate(p_state[::-1]):
            if idx == 0:
                continue

            ugate = Unitary(self._mct_generator(
                self.qubits + self.dim,
                self.qubits - 1 - pbit
            ))
            ugate | inc_mod(list(range(self.qubits + self.dim - 1, -1, -1)))
            X | inc_mod(self.qubits - 1 - pbit)

        for _0 in _0_state:
            X | inc_mod(self.qubits - 1 - _0)

        return inc_mod

    def _decre_mod(self):
        diff = (1 << self.qubits) - self.p_space
        bitl_diff = int(np.log2(diff)) + 1
        diff_state = [i for i in range(bitl_diff) if diff & (1 << i)]

        # Construct decrement mod cgate
        dec_mod = CompositeGate()
        X | dec_mod(self.qubits + self.dim - 1)
        for idx, dbit in enumerate(diff_state):
            if idx != 0:
                X | dec_mod(self.qubits - 1 - diff_state[idx - 1])

            ugate = Unitary(self._mct_generator(
                self.qubits + self.dim,
                self.qubits - 1 - dbit
            ))
            ugate | dec_mod(list(range(self.qubits + self.dim - 1, -1, -1)))

        X | dec_mod(self.qubits + self.dim - 1)
        for dbit in diff_state[:-1]:
            X | dec_mod(self.qubits - 1 - dbit)

        return dec_mod

    def circuit(self) -> Circuit:
        return self._circuit

    def run(self, device: str = "CPU"):            
        pass

    def visible(self):
        pass
