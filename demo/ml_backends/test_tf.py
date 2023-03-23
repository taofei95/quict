import tensorflow as tf
import numpy as np
import sys

sys.path.append("/home/zoker/quict")


from QuICT.simulation.state_vector import CircuitSimulator
from QuICT.core import Circuit
from QuICT.core.gate import *


def f(params):
    print(params)
    cir = Circuit(2)
    sim = CircuitSimulator()
    H | cir
    Rx(np.array(params)) | cir(0)
    sv = sim.run(cir)

    loss = sum(sv).real
    loss = tf.convert_to_tensor(loss)

    return loss


def vmap(f):
    def wrapper(*args, **kws):
        def pf(x):
            return f(x, *args[1:], **kws)

        return tf.vectorized_map(pf, args[0])

    return wrapper


def vvag(f):
    vf = vmap(f)

    def wrapper(*args, **kws):
        with tf.GradientTape() as tape:
            x = args[0]
            tape.watch(x)
            vs = vf(*args, **kws)
        grad = tape.gradient(vs, x)
        return vs, grad

    return wrapper


if __name__ == '__main__':
    vvag_f = vvag(f)

    param = np.array([3.0])
    param = tf.convert_to_tensor(param)
    
    f(param)

    opt = tf.keras.optimizers.Adam()
    # op = opt.minimize

    loss, grad = vvag_f(param)
    print(loss)
    print(grad)