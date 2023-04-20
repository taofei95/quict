import pickle
import numpy as np
import matplotlib.pyplot as plt

from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.utility.circuit_cost.quafu_backend import QuafuBackend

from my_tokens import quafu_api_token as my_token


def run_quafu():
    """
    Test Quafu Backend
    """
    bkd = QuafuBackend(api_token=my_token, system='ScQ-P136')
    circ = Circuit(2)
    H | circ(0)
    H | circ(1)

    print(bkd.execute_circuit(circ, n_shot=1000))


def get_benchmark():
    """
    Get 20 random circuit with #qubit <= 10, #gate <= 10
    """
    bkd = QuafuBackend(api_token=my_token, system='ScQ-P10')
    bmks = bkd.generate_benchmark(10, 10, 20)
    pickle.dump(bmks, open('bmks_10_10_20.pkl', 'wb'))


def analyze_data(filename):
    """
    Compare the cost and pst of the benchmark circuits
    """
    bmks = pickle.load(open(filename, 'rb'))
    bkd = QuafuBackend(api_token=my_token, system='ScQ-P10')

    data = []
    for idx, val in enumerate(bmks):
        circ, pst = val
        cost = bkd.estimated_cost(circ)
        data.append([pst, cost, idx])
        print([pst, cost, idx])
    data.sort()
    data = np.array(data)

    # for i in range(data.shape[0]):
    #     bmks[int(data[i, 2])][0].draw(filename=f'c{i}.jpg')

    plt.show()
    plt.plot(range(data.shape[0]), -np.log(data[:, 0]), label='pst-cost')
    plt.plot(range(data.shape[0]), data[:, 0], label='pst')
    plt.plot(range(data.shape[0]), data[:, 1], label='cost')
    plt.legend()
    plt.show()


def run_case():
    bkd = QuafuBackend(api_token=my_token, system='ScQ-P10')

    circ = Circuit(10)
    H | circ(0)
    H | circ(0)
    Rx(np.pi / 2) | circ(2)
    Ry(np.pi / 2) | circ(2)
    Ry(np.pi / 2) | circ(2)
    Ry(np.pi / 2) | circ(3)
    print(bkd._get_circuit_pst(circ))

    circ = Circuit(10)
    CZ | circ([5, 6])
    Ry(np.pi / 2) | circ(5)
    Ry(np.pi / 2) | circ(5)
    H | circ(9)
    Rx(np.pi / 2) | circ(9)
    Ry(np.pi / 2) | circ(9)
    print(bkd._get_circuit_pst(circ))

    circ = Circuit(10)
    Ry(np.pi / 2) | circ(8)
    Ry(np.pi / 2) | circ(8)
    CZ | circ([7, 8])
    H | circ(7)
    print(bkd._get_circuit_pst(circ))


if __name__ == '__main__':
    run_quafu()
