from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian


def maxcut_hamiltonian(edges: list, weights: list = None):
    if weights is not None:
        assert len(edges) == len(weights)
    pauli_list = []
    for edge in edges:
        pauli_list.append([-1.0, "Z" + str(edge[0]), "Z" + str(edge[1])])
    hamiltonian = Hamiltonian(pauli_list)

    return hamiltonian


def tsp_hamiltonian(nodes: int, weighted_edges: list):
    require_list = []
    penalty_list1 = []
    penalty_list2 = []
    for edge in weighted_edges:
        i = edge[0]
        j = edge[1]
        w = edge[2]
        for t in range(nodes - 1):
            # require_list.append([1 / 4, "I"])
            # require_list.append([-1 / 4, "I"])
            idx1 = i * nodes + t
            idx2 = j * nodes + t + 1
            require_list.append([w / 4, "Z" + str(idx1), "Z" + str(idx2)])
            idx1 = i * nodes + t + 1
            idx2 = j * nodes + t
            require_list.append([w / 4, "Z" + str(idx1), "Z" + str(idx2)])
        require_list.append(
            [w / 4, "Z" + str(i * nodes), "Z" + str((j + 1) * nodes - 1)]
        )
        require_list.append(
            [w / 4, "Z" + str(j * nodes), "Z" + str((i + 1) * nodes - 1)]
        )
    print(require_list)
    return


if __name__ == "__main__":
    nodes = 3
    edges = [[0, 1, 48], [1, 2, 63], [2, 0, 91]]
    tsp_hamiltonian(nodes, edges)
