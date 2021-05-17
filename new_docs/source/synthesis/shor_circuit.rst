Shor Circuit
======================

We implemented two circuits implementing Shor's algorithm, including **HRSShorFactor** and **BEAShorFactor**。The two circuit differ in the implementation of the **multiplying circuit** of **order-finding**.

Usage
-----------

The shor circuits are located in **QuICT.algorithm.XXX** modules, 
where **XXX** is submodules stands for different shor circuits, 
that is, it can be **HRSShorFactor** and **BEAShorFactor**.
The typical way to import a circuit would be like (take **HRSShorFactor** as example):

.. code-block:: python
    :linenos:

    from QuICT.algorithm import HRSShorFactor


After imported, the circuits in the module could be used. The typical way to use them would be like the (take **HRSShorFactor** as example):

.. code-block:: python
    :linenos:

    HRSShorFactor.run(1019 * 1021)

QuICT.algorithm.HRSShorFactor
------------------------------

    Bases: QuICT.algorithm._algorithm.Algorithm

implementation of shor algorithm.

QuICT.algorithm.HRSShorFactor.run(N)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

run the algorithm with fidelity Args:
    
    Args:
        N(int): the number to be factored
    
    Returns:
        int: a factor of N

QuICT.algorithm.BEAShorFactor
------------------------------

    Bases: QuICT.algorithm._algorithm.Algorithm

implementation of shor algorithm, which uses the one controlling-qubit trick. Notice that the algorithm uses only 2\*n+3 bits, but the circuit runs slower.  

QuICT.algorithm.BEAShorFactor.run(N)
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    Args:
        N(int): the number to be factored
    
    Returns:
        int: a factor of N
