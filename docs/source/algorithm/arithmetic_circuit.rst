Arithmetic Circuit
======================

We implemented several arithmetic circuits , 
including **Add**, **AddMod**, **MulMod**, **ExpMod** and **Division with remainder**.
Each aithmetic may have different realizations based on different designs,
users should decide which to use according to the description of each circuits.

Usage
-----------
The arithmetic circuits are located in **QuICT.qcda.synthesis.arithmetic.XXX** modules, 
where **XXX** is submodules stands for different operations or designs, 
that is, it can be operation names or designer names.
The typical way to import a circuit would be like (take **division** as example):

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.division import * 
    #RestoringDivision is included in module division

or 

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.tmvh import * 
    #RestoringDivision is also included in module tmvh

After imported, the circuits in the module could be used, like ordinary gates, 
by being applied on certain quregs with operator **|**. 
The typical way to use them would be like the (take **division** as example):

.. code-block:: python
    :linenos:

    circuit = Circuit(3*n + 1)
    r_q = circuit([i for i in range(n)])
    a_q = circuit([i for i in range(n,2*n)])
    b_q = circuit([i for i in range(2*n,3*n)])
    of_q = circuit(3*n)

    #After some procedures, a_q, b_q and r_q are now in arbitary states.
    RestoringDivision.execute(n) | (a_q,b_q,r_q,of_q)

TMVH module
--------------
In **QuICT.qcda.synthesis.arithmetic.tmvh**, we have **RippleCarryAdder**, **Multiplication**, **RestoringDivision**. 
These circuits are designed for ordinary usage (while the following designs for Shor usage).
Circuits are based on theses - https://arxiv.org/abs/1706.05113v1, https://arxiv.org/abs/1809.09732v1.

RippleCarryAdder
>>>>>>>>>>>>>>>>>

**RippleCarryAdder.execute(n)** takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.
The circuit leaves out the overflow bit. Qureg **a** keeps unchanged, the result is stored in qureg **b**.

:math:`|a,b\rangle \rightarrow |a,a+b\rangle`

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.tmvh import *

    circuit = Circuit(n * 2)
    qreg_a = circuit([i for i in range(n)])
    qreg_b = circuit([i for i in range(n, n * 2)])

    #After some procedures, the quregs are now in arbitary states.
    RippleCarryAdder.execute(n) | (qreg_a,qreg_b)

Multiplication
>>>>>>>>>>>>>>>>>

**Multiplication.execute(n)** takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.
The circuit takes two n-bit quantum inputs, stores the product in 2n-bit Qureg **p**, in need of one-bit **ancilla**.

:math:`|a,b,p=0,ancilla=0\rangle \rightarrow |a,b,a*b,0\rangle`

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.tmvh import *

    circuit = Circuit(4*n + 1)
    qreg_a = circuit([i for i in range(n)])
    qreg_b = circuit([i for i in range(n, 2 * n)])
    qreg_p = circuit([i for i in range(2*n, 4*n)])
    ancilla = circuit(4*n)

    #After some procedures, the qreg_a and qreg_b are now in arbitary states.
    Multiplication.execute(n) | (qreg_a,qreg_b,qreg_p,ancilla)

RestoringDivision
>>>>>>>>>>>>>>>>>

**RestoringDivision.execute(n)** takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.
The circuit takes two n-bit quantum inputs, stores the quotient in n-bit Qureg **a**, the remainder in n-bit Qureg **r**, in need of one-bit ancilla **overflow**.

:math:`|a,b,r=0,overflow=0\rangle \rightarrow |a\%b,b,a//b,0\rangle`

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.tmvh import *

    circuit = Circuit(3 * n + 1)
    qreg_a = circuit([i for i in range(n)])
    qreg_b = circuit([i for i in range(n, 2 * n)])
    qreg_r = circuit([i for i in range(2 * n, 3 * n)])
    overflow = circuit(3 * n)

    #After some procedures, the qreg_a and qreg_b are now in arbitary states.
    RestoringDivision.execute(n) | circuit

..
    VBE module
    --------------
    In **QuICT.qcda.synthesis.arithmetic.vbe**, we have **VBEAdder**, **VBEAdderMod**, **VBEMulAddMod** and **VBEExpMod**.
    These circuits are designed more for Shor usage than general arithmetic purpose.

    VBEAdder
    >>>>>>>>>>>>>>>>>

    **VBEAdder.execute(n)** constructs a circuit which adds two integers. 
    It takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.

    Qureg **a** keeps unchanged, the result is stored in qureg **b**,
    qureg **c** is clean ancilla, qubit **overflow** flips if the addition produces overflow. 

    :math:`|a,b,c=0,overflow\rangle \rightarrow |a,a+b,c=0,overflow'\rangle`

    .. code-block:: python
        :linenos:

        from QuICT.qcda.synthesis.arithmetic.vbe import *

        circuit = Circuit(3*n + 1)
        a_q = circuit([i for i in range(n)])
        b_q = circuit([i for i in range(n, 2*n)])
        c_q = circuit([i for i in range(2*n, 3*n)])
        overflow_q = circuit(3*n)

        #After some procedures, the quregs are now in arbitary states.
        VBEAdder.execute(n) | (a_q,b_q,c_q,overflow_q)

    VBEAdderMod
    >>>>>>>>>>>>>>>>>

    **VBEAdderMod.execute(N,n)** constructs a circuit which can add two integers module N. 
    It takes **N** as the constant modulus embedded in the structure of the circuit. 
    It takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.

    Qureg **a** keeps unchanged, the result is stored in qureg **b**,
    qureg **c**, **N_q**, **overflow** and **t** are clean ancilla. 

    :math:`|a,b,c=0,overflow=0,N_q=0,t=0\rangle \rightarrow |a,(a+b)mod N,c=0,overflow,N_q,t\rangle`

    .. code-block:: python
        :linenos:

        from QuICT.qcda.synthesis.arithmetic.vbe import *

        circuit = Circuit(4*n + 2)
        a_q = circuit([i for i in range(n)])
        b_q = circuit([i for i in range(n, 2*n)])
        c_q = circuit([i for i in range(2*n, 3*n)])
        overflow_q = circuit(3*n)
        N_q = circuit([i for i in range(3*n + 1, 4*n + 1)])
        t_q = circuit(4*n + 1)

        #After some procedures, the quregs are now in arbitary states.
        VBEAdderMod.execute(n,N) | (a_q,b_q,c_q,overflow_q,N_q,t_q)

    VBEMulAddMod
    >>>>>>>>>>>>>>>>>

    **VBEMulAddMod.execute(a,N,n,m)** constructs a circuit which computes multiplication-addition module N. 
    It takes **a** as a constant multiplier embedded in the structure of the circuit.
    It takes **N** as the constant modulus embedded in the structure of the circuit. 
    It takes **n** as the parameter indicating the length of **N**, to tailor the circuit to proper size.
    It takes **m** as the parameter indicating the length of **x**, to tailor the circuit to proper size.

    Qureg **x** keeps unchanged, the result is stored in qureg **b**,
    qureg **a_q**, **c**, **N_q**, **overflow** and **t** are clean ancilla. 

        \|x,a_q=0,b,c=0,overflow=0,N_q=0,t=0> -> \|x,a_q,(a*x + b) mod N,c,overflow,N_q,t>

    .. code-block:: python
        :linenos:

        from QuICT.qcda.synthesis.arithmetic.vbe import *

        circuit = Circuit(4*n + m + 2)
        x_q = circuit([i for i in range(m)])
        a_q = circuit([i for i in range(m,n + m)])
        b_q = circuit([i for i in range(n + m, 2*n + m)])
        c_q = circuit([i for i in range(2*n + m, 3*n + m)])
        overflow_q = circuit(3*n + m)
        N_q = circuit([i for i in range(3*n + m + 1, 4*n + m + 1)])
        t_q = circuit(4*n + m + 1)

        #After some procedures, the quregs are now in arbitary states.
        VBEMulAddMod.execute(a,N,n,m) | (x_q,a_q,b_q,c_q,overflow_q,N_q,t_q)

    VBEExpMod
    >>>>>>>>>>>>>>>>>

    **VBEExpMod.execute(a,N,n,m)** constructs a circuit which computes exponentiation module N. 
    It takes **a** as a constant base number embedded in the structure of the circuit.
    It takes **N** as the constant modulus embedded in the structure of the circuit. 
    It takes **n** as the parameter indicating the length of **N**, to tailor the circuit to proper size.
    It takes **m** as the parameter indicating the length of **x**, to tailor the circuit to proper size.

    Qureg **x** keeps unchanged, the result is stored in qureg **r**,
    qureg **a_q**, **c**, **N_q**, **overflow** and **t** are clean ancilla. 

        \|x,r=0,a_q=0,b=0,c=0,overflow=0,N_q=0,t=0> -> \|x,(a^x) mod N,a_q,b,c,overflow,N_q,t>

    .. code-block:: python
        :linenos:

        from QuICT.qcda.synthesis.arithmetic.vbe import *

        circuit = Circuit(m + 5 * n + 2)
        x_q = circuit([i for i in range(m)])
        r_q = circuit([i for i in range(m,n + m)])
        a_q = circuit([i for i in range(n + m, 2*n + m)])
        b_q = circuit([i for i in range(2*n + m, 3*n + m)])
        c_q = circuit([i for i in range(3*n + m, 4*n + m)])
        overflow_q = circuit(4*n + m)
        N_q = circuit([i for i in range(4*n + m + 1, 5*n + m + 1)])
        t_q = circuit(5*n + m + 1)

        #After some procedures, the quregs are now in arbitary states.
        VBEExpMod.execute(a,N,n,m) | (x_q,r_q,a_q,b_q,c_q,overflow_q,N_q,t_q)

BEA module
--------------
In **QuICT.qcda.synthesis.arithmetic.bea**, we have **BEAAdder**, **BEAAdderWired**, **BEAAdderWiredCC**, **BEAAdderMod**, **BEAMulMod**. 
Besides,there are a few circuits used as intermediate implementation of Shor's algorithm, which are not listed in the doc, but still tested and can be used.
These circuits are designed more for Shor usage than general arithmetic purpose.

BEAAdder
>>>>>>>>>>>>>>>>>

**BEAAdder.execute(n)** behaves like **VBEAdder.execute(n)**, but without Control and Overflow bits. 
It takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.

Qureg **a** keeps unchanged, the result is stored in qureg **b**.

:math:`|a,b\rangle \rightarrow |a,a+b\rangle`

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(n * 2)
    qreg_a = circuit([i for i in range(n)])
    qreg_b = circuit([i for i in range(n, n * 2)])

    #After some procedures, qreg_a and qreg_b are now in arbitary states.
    BEAAdder.execute(n) | (qreg_a,qreg_b)

BEAAdderWired
>>>>>>>>>>>>>>>>>

**BEAAdderWired.execute(n,a)** behaves like **BEAAdder.execute(n)**, but `a` is wired. `b` use n+1 bits to store, therefore guarantee no overflow.

:math:`|b\rangle \rightarrow |a+b\rangle`

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(n + 1)
    qreg_b = circuit([i for i in range(n + 1)])

    #After some procedures, the quregs are now in arbitary states.
    BEAAdderWired.execute(n,a) | qreg_b

..
    CCBEAAdderWired
    >>>>>>>>>>>>>>>>>

    **CCBEAAdderWired.execute(n,a)** is **BEAAdderWired.execute(n,a)** with 2 control bits.

    \|b,c> -> \|(c==0b11)?a+b:b,c>

    .. code-block:: python
        :linenos:

        from QuICT.qcda.synthesis.arithmetic.bea import *

        circuit = Circuit(n + 3)
        qreg_b = circuit([i for i in range(n + 1)])
        qreg_c = circuit([i for i in range(n + 1, n + 3)])

        #After some procedures, the quregs are now in arbitary states.
        CCBEAAdderWired.execute(n,a) | (qreg_b,qreg_c)

BEAAdderMod
>>>>>>>>>>>>>>>>>

**BEAAdderMod.execute(n,a,N)** constructs a circuit which can add two integers module N, and `a` is wired. 
It takes **N** as the constant modulus embedded in the structure of the circuit. 
It takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.

(Qureg): the qureg stores b, length is n+1,
low(Qureg):  the clean ancillary qubit, length is 1,

Qureg **b** stores result, guarantee no overflow,
qureg **low** is  the clean ancillary qubit. 

:math:`|b,low\rangle \rightarrow |(a+b)\%N,low\rangle`

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(n + 2)
    qreg_b = circuit([i for i in range(n + 1)])
    qreg_low = circuit([i for i in range(n + 1, n + 2)])

    #After some procedures, the quregs are now in arbitary states.
    BEAAdderMod.execute(n,a,N) | (qreg_b,qreg_low)

BEAMulMod
>>>>>>>>>>>>>>>>>

**BEAMulMod.execute(n,a,N)** constructs a circuit which computes multiplication-addition module N. 
It takes **a** as a constant multiplier embedded in the structure of the circuit.
It takes **N** as the constant modulus embedded in the structure of the circuit. 
It takes **n** as the parameter indicating the length of **N** and **x**, to tailor the circuit to proper size.

Qureg **x** keeps unchanged, the result is stored in qureg **b**,
qureg **low** is the clean ancillary qubit. 

:math:`|b,x,low\rangle \rightarrow |(b+ax)\%N,x,low\rangle`

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(2 * n + 2)
    qreg_b = circuit([i for i in range(n + 1)])
    qreg_x = circuit([i for i in range(n + 1, 2 * n + 1)])
    qreg_low = circuit(2 * n + 1)

    #After some procedures, the quregs are now in arbitary states.
    BEAMulMod.execute(n,a,N) | (qreg_b,qreg_x,qreg_low)

Performance indices
----------------------------

Here we list the performance indices of most circuits.

Addition circuits
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. csv-table::
 :header: "Circuit", "Description", "Qubit", "Size"
 :widths: 15, 30, 10, 10

 RippleCarryAdder,  ":math:`|a,b\rangle \rightarrow |a,a+b\rangle`",       :math:`2n`,     ":math:`7n`"
 BEAAdder,  ":math:`|a,b\rangle \rightarrow |a,a+b\rangle`",       :math:`2n`,     ":math:`\frac{3}{2}n^2`"
 HRSAdder, ":math:`|x,a_1,a_2\rangle \rightarrow |x+c,a_1,a_2\rangle`",       :math:`n+2`,     ":math:`25n\log{n}`"


Multiplication circuits
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. csv-table::
 :header: "Circuit", "Description", "Qubit", "Size"
 :widths: 15, 30, 10, 10

 Multiplication,  ":math:`|a,b,p=0,ancilla=0\rangle \rightarrow |a,b,a*b,0\rangle`",       :math:`4n+1`,     ":math:`7n^2`"
 BEAMulMod,  ":math:`|b,x,c,low\rangle \rightarrow |(b+ax) mod N,x,c,low\rangle`",       :math:`2n+3`,     ":math:`\frac{21}{2}n^3`"
 HRSMulMod, ":math:`|x,anc,ind\rangle \rightarrow |(ax) mod N,anc,ind\rangle`",       :math:`2n+1`,     ":math:`50n^2\log{n}`"

Division circuits
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. csv-table::
 :header: "Circuit", "Description", "Qubit", "Size"
 :widths: 15, 30, 10, 10

 RestoringDivision,  ":math:`|a,b,r=0,overflow=0\rangle \rightarrow |a\%b,b,a//b,0\rangle`",       :math:`3n`,     ":math:`14n^2`"

Exponentiation circuits
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

.. csv-table::
 :header: "Circuit", "Description", "Qubit", "Size"
 :widths: 15, 30, 10, 10

 BEAExpMod,  ":math:`|b,x,c,low\rangle \rightarrow |a^x mod N,x,c,low\rangle`",       :math:`2n+m+2`,     ":math:`\frac{21}{2}mn^3`"
 HRSExpMod, ":math:`|x,anc,ind\rangle \rightarrow |(a^x) mod N,anc,ind\rangle`",       :math:`2n+m+1`,     ":math:`50mn^2\log{n}`"
