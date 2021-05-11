Unitary Transform
===================
As is known that any quantum circuit on :math:`n` qubits corresponds to a unitary matrix
:math:`U\in SU(2^n)`. Unitary transform is a model that transforms a given unitary 
matrix :math:`U\in SU(2^n)` to a :math:`n`-qubit quantum circuit which only contains
:math:`1`-qubit and CNOT gates.

Example
-------------------
Here is a simple usage of this model.

.. code-block:: python
    :linenos:

    from scipy.stats import unitary_group
    
    from QuICT.core import *
    from QuICT.qcda.synthesis.unitary_transform import *
    
    if __name__ == '__main__':
        U = unitary_group.rvs(2 ** 3)
        compositeGate, _ = UTrans(U)
    
        circuit = Circuit(3)
        circuit.set_exec_gates(compositeGate)
        circuit.draw_photo(show_depth=False)

Function **unitary_group.rvs** returns a random unitary matrix. Here we generates a random 
:math:`8\times 8` matrix :math:`U\in SU(2^3)`, and transforms it to a :math:`3`-qubit 
quantum circuit.

.. figure:: ./images/ut_0.jpg

In the result figure above, custom gates are :math:`1`-qubit gates defined by certain
:math:`SU(2)` matrices.

The **_** in the return is preserved for possible global phase, Check *Optional Parameters* for 
more detail.

Result
-------------------
The algorithm implemented here would return a **CompositeGate** with :math:`\frac{23}{48}4^n
-\frac{3}{2}2^n+\frac{4}{3}` CNOT gates and some :math:`1`-qubit gates.([1], Table 1) The lower 
bound of the number of CNOT gates is proved to be :math:`\frac{1}{4}(4^n-n-1)`.([4], Prop. 4.1)

Principle
-------------------
The transform process mainly consists of two parts:

First, we recursively decompose the unitary matrix to CNOT gates, :math:`1`-qubit gates and 
:math:`SU(4)` matrices with quantum Shannon decomposition and decomposition of 
multiplexed-:math:`R_{y,z}` gates.([1], Thm. 8, 10, 13)

Then, we decompose the remaining :math:`SU(4)` matrices to :math:`3` CNOT gates and some 
:math:`1`-qubit gates with Cartan KAK decomposition.([2], 3.2)

After that, we revise the process of decomposition of multiplexed-:math:`R_y` gates ([1], 
Appendix A) and Cartan KAK decomposition ([3], Prop. IV.3, V.2) to reduce the coefficient 
of :math:`4^n` in the number of CNOT gates from :math:`\frac{9}{16}` to :math:`\frac{23}{48}`. 

Optional Parameters
-------------------
Despite of the matrix to be transformed, this model also provides some optional parameters for
researchers of interest in this method.

**include_phase_gate**: Decide whether to retain the global phase as a phase gate produced in 
the transform process, default `True`.(Then the global phase in **return** would be **None**, 
as is in the *Example*. Otherwise the **return** would be a tuple containing the **CompositeGate** 
and a **complex** which is the global phase.)

**mapping**: Specify the qubits and their order implied in the matrix to be transformed, which 
is a list of their labels from top to bottom, default `None`.(Then the order is :math:`0, 1,\dots, 
n-1` where :math:`n` is decided by the matrix.)

**recursive_basis**: Stop the recursive decomposition process at :math:`1` or :math:`2`-qubit
gates, default :math:`2`.(Then the next step would be Cartan KAK decomposition.)

Reference
-------------------
[1] https://arxiv.org/abs/quant-ph/0406176

[2] https://arxiv.org/abs/0806.4015

[3] https://arxiv.org/abs/quant-ph/0308033

[4] https://ieeexplore.ieee.org/document/1269020
