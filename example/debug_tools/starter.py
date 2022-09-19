# exec(open("build/starter.py").read())
from QuICT.simulation.state_vector.cpu_simulator import CircuitSimulator
from QuICT.simulation.state_vector.gpu_simulator import ConstantStateVectorSimulator
from QuICT.core import Circuit, circuit
from QuICT.core.gate import *
import QuICT

print(f"using {QuICT.__file__} ...")
import cupy as cp


def formatted_result(result, l, qregs, qregs_name):
    s = f"{'idx':{l+len(qregs)}}|{'prob':5}|{'phase/2pi':10}\n"
    s += f"qregs: {qregs_name}\n"
    for item in result:
        s += f"{item[0]}|{item[1]:5.3f}|{item[2]:10.3f}\n"
    return s


def formatted_qreg_slice(s, qreg):
    return "".join([s[idx] for idx in qreg])


def formatted_qreg(s, qregs):
    return "".join(["|" + formatted_qreg_slice(s, qreg) for qreg in qregs]) + "|"


def amp2idx(amp, show_n=5, formatted=True, qregs=None, qreg_names=None):
    """peek top-n probability result in amp

    Args:
        amp (cupy.ndarray): amplitude
        show_n (int, optional): number of shown results. Defaults to 5.
        formatted (bool, optional): Set to True unless you need the original result. Defaults to True.
        qregs (list<list<int>>, optional): list of qreg indices. Defaults to None.
        qreg_names (list<str>, optional): list of qreg names. Defaults to None.

    Returns:
        str | list<string,float,float>: the formatted result's string or,
        if formatted=False, list of 3-tuple that contains (measurement-result, probability, phase/2pi)
    """
    from math import log2

    n_bit = int(log2(len(amp)))
    if qregs is None:
        qregs = [list(range(n_bit))]

    amp = cp.asnumpy(amp)
    pr = np.power(np.abs(amp), 2)
    ps = np.real(np.log(amp) / (2j * np.pi))
    arg = np.argsort(pr)[::-1]
    result = []
    for i in range(show_n):
        result.append(
            (
                formatted_qreg(bin(arg[i])[2:].rjust(n_bit, "0"), qregs),
                pr[arg[i]],
                ps[arg[i]],
            )
        )
    if formatted:
        return formatted_result(result, n_bit, qregs, qreg_names)
    else:
        return result


def peeeeeeek(n, qregs=None):
    """peek top-n probability result. external Variable `amp` needed.

    Args:
        n (int): _description_
        qregs (list<list<int>>, optional): list of qreg indices. Defaults to None.
    """
    print(amp2idx(amp, show_n=n, qregs=qregs))


def trace_prob(amp, c):
    """diag(partial_trace(|amp><amp|))

    Args:
        amp (cupy.ndarray): amplitude
        c (list<int>): qubits preserved

    Returns:
        numpy.ndarray: probability distribution for subspace on qubits preserved
    """
    from math import log2

    amp = cp.asnumpy(amp)
    n = int(log2(len(amp)))
    m = len(c)
    c_ortho = list(set(list(range(n))) - set(c))
    assert (1 << n) == len(amp)

    prob = np.power(np.abs(amp), 2)
    new_prob = np.sum(np.reshape(prob, tuple([2 for i in range(n)])), tuple(c_ortho))
    return np.reshape(new_prob, (1 << m,))


def set_qureg(qreg_index, N):
    """set qureg to the state |N> in big-endian, same direction as arithmetic circuits

    Args:
        qreg_index (list<int>): _description_
        N (int): _description_

    Returns:
        CompositeGate: a gate converts |0> to |N>
    """
    gate_set = CompositeGate()
    n = len(qreg_index)
    with gate_set:
        for i in range(n):
            if N % 2 == 1:
                X & qreg_index[n - 1 - i]
            N = N // 2
    return gate_set


helpers = {formatted_result, amp2idx, peeeeeeek, trace_prob, set_qureg}

print("using helpers:")
for helper in helpers:
    print("\t" + helper.__name__)

############
# EXTERNAL #
############

############
#   TEST   #
############

############
# UNITTEST #
############
