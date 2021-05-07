import numpy as np
from QuICT import Circuit, H, X, Measure, PermFx
from QuICT.qcda.synthesis.initial_state_preparation import InitialStatePreparation
from QuICT.qcda.synthesis.mct import MCTOneAux

def deutsch_jozsa_main_oracle(f, qreg, ancilla):
    PermFx(f) | (qreg, ancilla)


def weight_decison_para(n, k, l):
    kap = k / n
    lam = l / n
    for d in range(n + 2):
        for gamma in range(1, 2 * d - 2):
            s = (1 - np.cos(gamma * np.pi / (2 * d - 1))) / 2
            t = (1 - np.cos((gamma + 1) * np.pi / (2 * d - 1))) / 2
            if lam * s >= kap * t and (1 - kap) * (1 - t) >= (1 - lam) * (1 - s):
                a = np.sqrt((l - k) / (t - s) - (l * s - k * t) / (t - s) - n)
                b = np.sqrt(abs(l * s - k * t) / (t - s))
                return d, gamma, a, b


def run_weight_decision(f, n, k, l, oracle):
    '''
    自定义oracle,对一个函数f使用kl算法判定
    :param f:       待判定的函数
    :param n:       待判定的函数的输入长度
    :param k:       较小的可能的权重
    :param l:       较大的可能的权重
    :param oracle:  oracle函数
    '''
    num = int(np.ceil(np.log2(n + 2))) + 2
    # Determine number of qreg
    circuit = Circuit(num)
    d, gamma, a, b = weight_decison_para(n, k, l)
    # start the eng and allocate qubits
    qreg = circuit([i for i in range(num - 2)])

    ancilla = circuit(num - 2)
    empty = circuit(num - 1)
    # Start with qreg in equal superposition and ancilla in |->
    N = np.power(2, num - 2)
    value = [0 for _ in range(N)]
    for i in range(n):
        value[i] = 1 / np.sqrt(n + a ** 2 + b ** 2)
    value[N - 2] = a / np.sqrt(n + a ** 2 + b ** 2)
    value[N - 1] = b / np.sqrt(n + a ** 2 + b ** 2)

    # Apply oracle U_f which flips the phase of every state |x> with f(x) = 1
    InitialStatePreparation.execute(value) | qreg
    X | ancilla
    H | ancilla

    for i in range(d - 1):
        oracle(f, qreg, ancilla)
        MCTOneAux | circuit

        InitialStatePreparation.execute(value) ^ qreg
        X | qreg
        MCTOneAux | circuit
        X | qreg
        InitialStatePreparation.execute(value) | qreg

    # Apply H,X to recover ancilla
    H | ancilla
    X | ancilla
    oracle(f, qreg, ancilla)
    MCTOneAux | circuit

    # Measure
    Measure | qreg
    Measure | ancilla

    circuit.exec()

    y = int(qreg)
    if int(ancilla) == gamma % 2:
        print('Weight is %d' % (k))
    else:
        print('Weight is %d' % (l))

if __name__ == '__main__':
    test_number = 2
    test = [1, 1, 1, 0, 0, 0, 0, 0]
    run_weight_decision(test, 3, 1, 3, deutsch_jozsa_main_oracle)
