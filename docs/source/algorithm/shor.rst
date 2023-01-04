Shor
====

Shor's algorithm factors integers in polynomial time, which provides an
exponential acceleration over the best-known classical algorithm. Its
existence suggests a serious threat to cryptography system based on
computational security of integer factorization, such as the widely used
RSA system.


Shor's algorithm is related with another problem called *period
finding*. Detailed explanation can be found in other sources [1]_ [2]_
and thus skipped. 


The most costly component in Shor's circuit is the module for exponentiation, 
which is exquisitely designed to save resources. The detailed design can be found
in docs for arithmetic circuits. And the two implementations of Shor's algorithm is
given in ``BEA_shor`` and ``HRS_shor``. The only difference is in the
:math:`controlled-U_a` circuit part. The former one is smaller in
circuit depth while the latter is smaller in circuit width.

The required resources in the two implementations are shown in table

Required Resources 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. csv-table::
 :header: "Circuit", "Qubit", "Size"
 :widths: 15, 10, 10

 BEA_shor,         :math:`2n+3`,     ":math:`21n^4`"
 HRS_shor,        :math:`2n+2`,     ":math:`100n^3\log{n}`"

BEA Shor
--------

The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
St´ephane Beauregard in ``Circuit for Shor’s algorithm using 2n+3 qubits``\

HRS Shor
--------

The (2n+2)-qubit circuit used in the Shor algorithm is designed by
THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in
``Factoring using 2n+2 qubits with Toffoli based modular multiplication``.

Example
-------

Following is a demonstration of how to use ``BEA_shor`` and the submodule ``order_finding``.

.. code:: python

    from QuICT.algorithm import HRSShorFactor, BEAShorFactor

    N = int(input("[HRS]Input the number to be factored: "))
    a = HRSShorFactor.run(N,5,'demo')
    print("HRSShor found factor", a)

    N = int(input("[BEA]Input the number to be factored: "))
    a = BEAShorFactor.run(N,5,'demo')
    print("BEAShor found factor", a)

.. code:: python

    from QuICT.algorithm import HRS_order_finding, BEA_order_finding

    # N = int(input("Input the modulo N: "))
    # a = int(input("Input the element wanting the order: "))

    print('HRS order finding')
    order = HRS_order_finding.run(3,11,'demo')
    if order != 0:
        print("HRS_order_finding found order", order)

    print('BEA order finding')
    order = BEA_order_finding.run(3,11,'demo')
    if order != 0:
        print("BEA_order_finding found order", order)

.. [1]
   Nielsen, M. A., & Chuang, I. L. (2019). *Quantum computation and
   quantum information*. Cambridge Cambridge University Press.

.. [2]
   https://qiskit.org/textbook/ch-algorithms/shor.html
