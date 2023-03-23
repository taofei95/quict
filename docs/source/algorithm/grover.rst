Grover
======

Quantum computer has a great advantage over classical computer in its
superior speed searching databases. More specifically, Grover's
algorithm demonstrates this capability. This algorithm speed up an
unstructured search problem quadratically, and can serve as a subroutine
in a variety of other algorithms for improvement. Some variation of
Grover's algorithm, such as Partial Grover Search, provides a tradeoff
between running speed and searching precision.

Standard Grover Search
----------------------

Search algorithm with one target. The detailed explanation can be found
in QCQI [1]_.

Partial Grover Search
---------------------

Partial Grover Search with one target. See `this
paper <https://arxiv.org/abs/quant-ph/0407122>`__ for detailed
explanation.

Example
-------

Following is a demonstration of how to use ``grover``.

.. code:: python

   import pytest

   from QuICT.algorithm.quantum_algorithm.grover import Grover, PartialGrover, GroverWithPriorKnowledge
   from QuICT.core import *
   import QuICT
   print(QuICT.__file__)
   
   
   def main_oracle(f, qreg, ancilla):
       PermFx(f) | (qreg, ancilla)
   
   
   def calculate_base_vector(n: int, target: int):
       """ include alpha, beta, phi, theta
   
       Returns:
           tuple: alpha, beta, phi
       """
       N = 2**n
       M = 1
       alpha = np.ones(N)
       alpha[target] = 0.
       alpha = alpha/np.sqrt(N-M)
       beta = np.zeros(N)
       beta[target] = 1.
       beta = beta/np.sqrt(M)
       phi = np.ones(N)/np.sqrt(N)
       return alpha, beta, phi
   
   
   def test_grover():
       print(
           "standard grover demonstration: given f defined on [0,2**n-1], find the only x that f(x)=1:")
       n = int(input("\tinput space size n: "))
       target = int(input("\tinput target x: "))
       N = 2**n
       f = [0] * N
       f[target] = 1
       alpha, beta, phi = calculate_base_vector(n, target)
       result = Grover.run(f, n, main_oracle, demo_mode=True,
                           **{"alpha": alpha, "beta": beta, "phi": phi, "target": target})
       if target != result:
           print("Failed: For n = %d, target = %d, found = %d" %
                 (n, target, result))
       else:
           print("Succeed: found target")

.. [1]
   Nielsen, M. A., & Chuang, I. L. (2019). *Quantum computation and
   quantum information*. Cambridge Cambridge University Press.
