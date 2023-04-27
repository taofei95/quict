import pickle
import numpy as np
import matplotlib.pyplot as plt

from QuICT.core import *
from QuICT.core.gate import *
from QuICT.qcda.utility.circuit_cost.quafu_backend import QuafuBackend
from QuICT.simulation import Simulator

from my_tokens import quafu_api_token as my_token


def run_quafu():
    """
    Test Quafu Backend
    """
    bkd = QuafuBackend(api_token=my_token, system='ScQ-P10')
    circ = Circuit(2)
    Rx(np.pi) | circ(0)

    print(bkd._get_circuit_measured_fidelity(circ))
    # res = bkd.execute_circuit(circ, n_shot=1000)
    # print(res)


def get_benchmark():
    """
    Get 10 random circuit with #qubit <= 10, #gate <= 10
    """
    bkd = QuafuBackend(api_token=my_token, system='ScQ-P18')
    bmks = bkd.generate_benchmark(10, 10, 15)
    pickle.dump(bmks, open('bmks_scq18_10_10_15.pkl', 'wb'))


def analyze_data(filename):
    """
    Compare the cost and pst of the benchmark circuits
    """
    bmks = pickle.load(open(filename, 'rb'))
    bkd = QuafuBackend(api_token=my_token, system='ScQ-P18')

    data = []
    for idx, val in enumerate(bmks):
        circ, pst = val
        cost = bkd.estimated_cost(circ)
        data.append([pst, cost, idx])
        print([pst, cost, idx])
    data.sort()
    data = np.array(data)

    for i in range(data.shape[0]):
        bmks[int(data[i, 2])][0].draw(filename=f'c{i}.jpg')

    plt.show()
    plt.plot(range(data.shape[0]), -np.log(data[:, 0]), label='pst-cost')
    plt.plot(range(data.shape[0]), data[:, 0], label='pst')
    plt.plot(range(data.shape[0]), data[:, 1], label='cost')
    plt.legend()
    plt.show()


def run_case():
    bmks = pickle.load(open('bmks_scq18_10_10_15.pkl', 'rb'))
    bkd = QuafuBackend(api_token=my_token, system='ScQ-P18')
    circ, pst = bmks[6]
    print(pst)
    print(bkd._get_circuit_pst(circ))


def run_simulator():
    sim = Simulator()
    circ = Circuit(2)
    X | circ(0)
    res = sim.run(circ)
    print(res)


if __name__ == '__main__':
    # get_benchmark()
    # analyze_data('bmks_scq18_10_10_15.pkl')
    # run_simulator()
    run_case()
