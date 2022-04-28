Weight Decision
===================

The weight decision problem, which requires determining the Hamming weight of a given
binary string, is a natural and important problem, with applications in cryptanalysis,
coding theory, fault-tolerant circuit design, and so on. In this work, we consider a partial 
Boolean function which distinguishes whether the Hamming weight of the length-n input is k or 
it is l.

The algorithm design is based on `Exact quantum query complexity of weight decision problems via Chebyshev polynomials`__

.. __: https://arxiv.org/abs/1801.05717

How to use weight decision in QuICT
-----------------------------------

.. code-block:: python

    import numpy as np

    from QuICT.algorithm import WeightDecision
    from QuICT.core.gate import PermFx


    # Initial k, l value
    k = 6
    l = k + 1
    T = l + 1
    final = l if np.random.randint(0, 2) else k

    # Using WeightDecision algorithm to get weight
    flag = False
    oracle = PermFx(int(np.ceil(np.log2(T + 2))), [1, 2])
    for _ in range(5):
        ans = WeightDecision.run(T, k, l, oracle)
        if final == ans:
            flag = True
            break

    assert flag
