from .mcts_node import *
from .utility import *


def fill_adj(array: np.ndarray):
    out = np.zeros(len(array) - 1, dtype=array.dtype)
    for i in range(len(out)):
        if array[i + 1] == -1:
            out[i] = array[0]
        else:
            out[i] = array[i + 1]
    return out


class Cost:
    def __init__(self, size: int = 150, circuit: DAG = None, coupling_graph: CouplingGraph = None,
                 config: RLConfig = default_rl_config, model_file: str = None):
        self._coupling_graph = coupling_graph
        self._circuit = circuit
        self._config = config
        self._size = size

        num_of_gates, subcircuit = self._circuit.get_subcircuit(front_layer=circuit.front_layer,
                                                                qubit_mask=circuit.initial_qubit_mask,
                                                                num_of_gates=size, gates_threshold=size)
        self._padding_mask = np.zeros(size, dtype=np.uint8)
        self._padding_mask[num_of_gates:] = 1
        self._adj = np.apply_along_axis(fill_adj, axis=1,
                                        arr=np.concatenate([np.arange(0, size)[:, np.newaxis], subcircuit[:, 0:5]],
                                                           axis=1))
        self._qubits = subcircuit[:, 5:]
        self.num_of_gates = num_of_gates

    def __call__(self, qubit_mapping: np.ndarray):
        value = 0.0
        qubits = qubit_mapping[self._qubits]
        for i in range(self.num_of_gates):
            value += self._coupling_graph.distance(qubits[i, 0], qubits[i, 1])
        return value


def simulated_annealing(init_mapping: np.ndarray, cost: Cost, method: str = None, param: Dict = None) -> np.ndarray:
    result = []
    T_max = param["T_max"]
    T_min = param["T_min"]
    num_of_iterations = param["iterations"]
    t = T_max
    alpha = param["alpha"]
    valuebest = np.infty
    valuenew = 0
    valuecurrent = 0
    num = init_mapping.shape[0]
    new_mapping = init_mapping.copy()
    cur_mapping = init_mapping.copy()
    best_mapping = init_mapping.copy()
    while t > T_min:
        for i in np.arange(num_of_iterations):
            if np.random.rand() > 0.5:
                while True:
                    loc1 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    loc2 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    if loc1 != loc2:
                        break
                new_mapping[loc1], new_mapping[loc2] = new_mapping[loc2], new_mapping[loc1]
            else:
                while True:
                    loc1 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    loc2 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    loc3 = np.int(np.ceil(np.random.rand() * (num - 1)))

                    if (loc1 != loc2) & (loc2 != loc3) & (loc1 != loc3):
                        break
                if loc1 > loc2:
                    loc1, loc2 = loc2, loc1
                if loc2 > loc3:
                    loc2, loc3 = loc3, loc2
                if loc1 > loc2:
                    loc1, loc2 = loc2, loc1

                tmp = new_mapping[loc1: loc2].copy()
                new_mapping[loc1: loc3 - loc2 + 1 + loc1] = new_mapping[loc2: loc3 + 1]
                new_mapping[loc3 - loc2 + 1 + loc1: loc3 + 1] = tmp

            valuenew = cost(new_mapping)
            if valuenew < valuecurrent:
                valuecurrent = valuenew
                cur_mapping[:] = new_mapping

                if valuenew < valuebest:
                    valuebest = valuenew
                    best_mapping[:] = new_mapping
            else:
                if np.random.rand() < np.exp(-(valuenew - valuecurrent) / t):
                    valuecurrent = valuenew
                    cur_mapping[:] = new_mapping
                else:
                    new_mapping[:] = cur_mapping
        t = alpha * t
        result.append(valuebest)

    return result, best_mapping
