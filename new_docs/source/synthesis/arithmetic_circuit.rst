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
    RestoringDivision(n) | (a_q,b_q,r_q,of_q)

Next we will use VBE module to demonstrate more detailed usage.

VBE module
--------------
In **QuICT.qcda.synthesis.arithmetic.vbe**, we have **VBEAdder**, **VBEAdderMod**, **VBEMulAddMod** and **VBEExpMod**.

VBEAdder
>>>>>>>>>>>>>>>>>

**VBEAdder(n)** constructs a circuit which can add two integers. 
It takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.

Qureg **a** keeps unchanged, the result is stored in qureg **b**,
qureg **c** is clean ancilla, qubit **overflow** flips if the addition produces overflow. 

|a,b,c=0,overflow> -> |a,a+b,c=0,overflow'>

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.vbe import *

    circuit = Circuit(3*n + 1)
    a_q = circuit([i for i in range(n)])
    b_q = circuit([i for i in range(n, 2*n)])
    c_q = circuit([i for i in range(2*n, 3*n)])
    overflow_q = circuit(3*n)

    #After some procedures, the quregs are now in arbitary states.
    VBEAdder(n) | (a_q,b_q,c_q,overflow_q)

VBEAdderMod
>>>>>>>>>>>>>>>>>

**VBEAdderMod(N,n)** constructs a circuit which can add two integers module N. 
It takes **N** as the constant modulus embedded in the structure of the circuit. 
It takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.

Qureg **a** keeps unchanged, the result is stored in qureg **b**,
qureg **c**, **N_q**, **overflow** and **t** are clean ancilla. 

    |a,b,c=0,overflow=0,N_q=0,t=0> -> |a,(a+b) mod N,c=0,overflow,N_q,t>

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
    VBEAdderMod(n,N) | (a_q,b_q,c_q,overflow_q,N_q,t_q)

VBEMulAddMod
>>>>>>>>>>>>>>>>>

**VBEMulAddMod(a,N,n,m)** constructs a circuit which computes multiplication-addition module N. 
It takes **a** as a constant multiplier embedded in the structure of the circuit.
It takes **N** as the constant modulus embedded in the structure of the circuit. 
It takes **n** as the parameter indicating the length of **N**, to tailor the circuit to proper size.
It takes **m** as the parameter indicating the length of **x**, to tailor the circuit to proper size.

Qureg **x** keeps unchanged, the result is stored in qureg **b**,
qureg **a_q**, **c**, **N_q**, **overflow** and **t** are clean ancilla. 

    |x,a_q=0,b,c=0,overflow=0,N_q=0,t=0> -> |x,a_q,(a*x + b) mod N,c,overflow,N_q,t>

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
    VBEMulAddMod(a,N,n,m) | (x_q,a_q,b_q,c_q,overflow_q,N_q,t_q)

VBEExpMod
>>>>>>>>>>>>>>>>>

**VBEExpMod(a,N,n,m)** constructs a circuit which computes exponentiation module N. 
It takes **a** as a constant base number embedded in the structure of the circuit.
It takes **N** as the constant modulus embedded in the structure of the circuit. 
It takes **n** as the parameter indicating the length of **N**, to tailor the circuit to proper size.
It takes **m** as the parameter indicating the length of **x**, to tailor the circuit to proper size.

Qureg **x** keeps unchanged, the result is stored in qureg **r**,
qureg **a_q**, **c**, **N_q**, **overflow** and **t** are clean ancilla. 

    |x,r=0,a_q=0,b=0,c=0,overflow=0,N_q=0,t=0> -> |x,(a^x) mod N,a_q,b,c,overflow,N_q,t>

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
    VBEExpMod(a,N,n,m) | (x_q,r_q,a_q,b_q,c_q,overflow_q,N_q,t_q)

BEA module
--------------
In **QuICT.qcda.synthesis.arithmetic.bea**, we have **BEAAdder**, **BEAAdderWired**, **BEAAdderWiredCC**, **BEAAdderMod**, **BEAMulMod**. 
Besides,there are a few circuits used as intermediate implementation of Shor's algorithm, which are not listed in the doc, but still tested and can be used.

BEAAdder
>>>>>>>>>>>>>>>>>

**BEAAdder(n)** behaves like **VBEAdder(n)**, but without Control and Overflow bits. 
It takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.

Qureg **a** keeps unchanged, the result is stored in qureg **b**.

|a,b> -> |a,a+b>

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(n * 2)
    qreg_a = circuit([i for i in range(n)])
    qreg_b = circuit([i for i in range(n, n * 2)])

    #After some procedures, the quregs are now in arbitary states.
    BEAAdder(n) | (qreg_a,qreg_b)

BEAAdderWired
>>>>>>>>>>>>>>>>>

**BEAAdderWired(n,a)** behaves like **BEAAdder(n)**, but `a` is wired. `b` use n+1 bits to store, therefore guarantee no overflow.

|b> -> |a+b>

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(n + 1)
    qreg_b = circuit([i for i in range(n + 1)])

    #After some procedures, the quregs are now in arbitary states.
    BEAAdderWired(n,a) | qreg_b

BEAAdderWiredCC
>>>>>>>>>>>>>>>>>

**BEAAdderWiredCC(n,a)** is **BEAAdderWired(n,a)** with 2 control bits.

|b,c> -> |(c==0b11)?a+b:b,c>

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(n + 3)
    qreg_b = circuit([i for i in range(n + 1)])
    qreg_c = circuit([i for i in range(n + 1, n + 3)])

    #After some procedures, the quregs are now in arbitary states.
    BEAAdderWiredCC(n,a) | (qreg_b,qreg_c)

BEAAdderMod
>>>>>>>>>>>>>>>>>

**BEAAdderMod(n,a,N)** constructs a circuit which can add two integers module N, and `a` is wired. 
It takes **N** as the constant modulus embedded in the structure of the circuit. 
It takes **n** as the parameter indicating the length of the integer, to tailor the circuit to proper size.

(Qureg): the qureg stores b,        length is n+1,
low(Qureg):  the clean ancillary qubit, length is 1,

Qureg **b** stores result, guarantee no overflow,
qureg **low** is  the clean ancillary qubit. 

    |b,low> -> |(a+b)%N,low>

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(n + 2)
    qreg_b = circuit([i for i in range(n + 1)])
    qreg_low = circuit([i for i in range(n + 1, n + 2)])

    #After some procedures, the quregs are now in arbitary states.
    BEAAdderMod(n,a,N) | (qreg_b,qreg_low)

BEAMulMod
>>>>>>>>>>>>>>>>>

**BEAMulMod(n,a,N)** constructs a circuit which computes multiplication-addition module N. 
It takes **a** as a constant multiplier embedded in the structure of the circuit.
It takes **N** as the constant modulus embedded in the structure of the circuit. 
It takes **n** as the parameter indicating the length of **N** and **x**, to tailor the circuit to proper size.

Qureg **x** keeps unchanged, the result is stored in qureg **b**,
qureg **low** is the clean ancillary qubit. 

    |b,x,low> -> |(b+ax)%N,x,low>

.. code-block:: python
    :linenos:

    from QuICT.qcda.synthesis.arithmetic.bea import *

    circuit = Circuit(2 * n + 2)
    qreg_b = circuit([i for i in range(n + 1)])
    qreg_x = circuit([i for i in range(n + 1, 2 * n + 1)])
    qreg_low = circuit(2 * n + 1)

    #After some procedures, the quregs are now in arbitary states.
    BEAMulMod(n,a,N) | (qreg_b,qreg_x,qreg_low)
