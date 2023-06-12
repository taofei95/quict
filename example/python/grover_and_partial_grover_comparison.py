from QuICT.core.gate import *
from QuICT.core.gate.backend import MCTOneAux
from QuICT.algorithm.quantum_algorithm import Grover, PartialGrover
from QuICT.simulation.state_vector import StateVectorSimulator
import numpy as np

# ### Grover模块和PartialGrover模块的比较

# 为了比较Grover模块和PartialGrover模块，我们考虑只有一个解的情况。

# 有四个命题$x_i:\sum_j (1-[x_j])=i, i=1,2,3,4$，我们想要判断每一个命题的真假。请注意，这些命题相互矛盾，因此最多只有一个是正确的，那么很容易看出只有第三个命题为真。

# 更确切地说，$f(x)=\land_{i=1,2,3,4}((\neg x_i\land\sum_{j\neq i}(1-[x_j])\neq i) \lor (x_i\land\sum_{j\neq i}(1-[x_j])=i))$。但实际上我们用$f_a(x)=f(x)\land(\land_{i=4+1...4+a} x_i)$来区分Grover和Partial Grover模块。

# 同样，我们可以用Grover模块构建谕示电路并找到解决方案。

def sentence(control_q:int,local_var_q,clause_q:int,ancilla_q):
    assert len(local_var_q)==3 and len(ancilla_q)==2
    cgate = CompositeGate()
    with cgate:
        CX & [local_var_q[0],clause_q]
        CX & [local_var_q[1],clause_q]
        CX & [local_var_q[2],clause_q]
        CCX & [local_var_q[1],local_var_q[2],ancilla_q[0]]
        X & ancilla_q[0]
        CCX & [clause_q,ancilla_q[0],ancilla_q[1]]
        X & ancilla_q[0]
        #uncompute clause_q
        CX & [local_var_q[0],clause_q]
        CX & [local_var_q[1],clause_q]
        CX & [local_var_q[2],clause_q]

        X & ancilla_q[1]
        CCX & [control_q,ancilla_q[1],clause_q]
        X & ancilla_q[1]
        X & clause_q

        #uncompute ancilla_q
        X & ancilla_q[0]
        CCX & [local_var_q[0],ancilla_q[0],ancilla_q[1]]
        CCX & [local_var_q[1],ancilla_q[0],ancilla_q[1]]
        CCX & [local_var_q[2],ancilla_q[0],ancilla_q[1]]
        X & ancilla_q[0]
        CCX & [local_var_q[1],local_var_q[2],ancilla_q[0]]
    return cgate

def sentence_adjoint(control_q:int,local_var_q,clause_q:int,ancilla_q):
    assert len(local_var_q)==3 and len(ancilla_q)==2
    cgate = CompositeGate()
    with cgate:
        #uncompute ancilla_q
        CCX & [local_var_q[1],local_var_q[2],ancilla_q[0]]
        X & ancilla_q[0]
        CCX & [local_var_q[2],ancilla_q[0],ancilla_q[1]]
        CCX & [local_var_q[1],ancilla_q[0],ancilla_q[1]]
        CCX & [local_var_q[0],ancilla_q[0],ancilla_q[1]]
        X & ancilla_q[0]

        X & clause_q
        X & ancilla_q[1]
        CCX & [control_q,ancilla_q[1],clause_q]
        X & ancilla_q[1]

        CX & [local_var_q[2],clause_q]
        CX & [local_var_q[1],clause_q]
        CX & [local_var_q[0],clause_q]
        X & ancilla_q[0]
        CCX & [clause_q,ancilla_q[0],ancilla_q[1]]
        X & ancilla_q[0]
        CCX & [local_var_q[1],local_var_q[2],ancilla_q[0]]
        CX & [local_var_q[2],clause_q]
        CX & [local_var_q[1],clause_q]
        CX & [local_var_q[0],clause_q]
    return cgate

def CCCX(reg_q):
    assert len(reg_q)==5
    cgate = CompositeGate()
    with cgate:
        CCX & [reg_q[0],reg_q[1],reg_q[4]]
        H & [reg_q[3]]
        S & [reg_q[4]]
        CCX & [reg_q[2],reg_q[3],reg_q[4]]
        S_dagger & [reg_q[4]]
        CCX & [reg_q[0],reg_q[1],reg_q[4]]
        S & [reg_q[4]]
        CCX & [reg_q[2],reg_q[3],reg_q[4]]
        H & [reg_q[3]]
        S_dagger & [reg_q[4]]
    return cgate

def puzzle_oracle():
    n_true_var = 4
    n_var = 8
    n_clause = 4
    var_q = list(range(n_var))
    clause_q = list(range(n_var,n_var+n_clause))
    result_q = [n_var+n_clause]
    ancilla_q = [n_var+n_clause+1,n_var+n_clause+2]
    cgate = CompositeGate()
    with cgate:
        # # |-> in result_q
        X & result_q[0]
        H & result_q[0]
        # sentence1
        for i in [1,2,3]:
            X & i
        sentence(0,[1,2,3],clause_q[0],ancilla_q) | cgate
        for i in [1,2,3]:
            X & i
        # sentence2
        sentence(1,[0,2,3],clause_q[1],ancilla_q) | cgate
        # sentence3
        for i in [0,1,3]:
            X & i
        CCCX([0,1,3,clause_q[2],ancilla_q[1]]) | cgate
        X & clause_q[2]
        CCX & [2,clause_q[2],ancilla_q[0]]
        X & clause_q[2]
        CCCX([0,1,3,clause_q[2],ancilla_q[1]]) | cgate
        for i in [0,1,3]:
            X & i
        CX & [ancilla_q[0],clause_q[2]]
        CX & [clause_q[2],ancilla_q[0]]
        X & clause_q[2]
        # sentence4
        for i in [0,1,2,3]:
            X & i
        CCCX([0,1,2,ancilla_q[0],ancilla_q[1]]) | cgate
        X & ancilla_q[0]
        CCX & [3,ancilla_q[0],clause_q[3]]
        X & ancilla_q[0]
        CCCX([0,1,2,ancilla_q[0],ancilla_q[1]]) | cgate
        for i in [0,1,2,3]:
            X & i
        # MCT
        MCTOneAux().execute(n_var-n_true_var+len(clause_q)+2) & (var_q[n_true_var:n_var]+clause_q+result_q+[ancilla_q[0]])
        #uncompute clauses
        # |-> in result_q
        H & result_q[0]
        X & result_q[0]
        # sentence1
        for i in [1,2,3]:
            X & i
        sentence_adjoint(0,[1,2,3],clause_q[0],ancilla_q) | cgate
        for i in [1,2,3]:
            X & i
        # sentence2
        sentence_adjoint(1,[0,2,3],clause_q[1],ancilla_q) | cgate
        # sentence3
        X & clause_q[2]
        CX & [clause_q[2],ancilla_q[0]]
        CX & [ancilla_q[0],clause_q[2]]
        for i in [0,1,3]:
            X & i
        CCCX([0,1,3,clause_q[2],ancilla_q[1]]) | cgate
        X & clause_q[2]
        CCX & [2,clause_q[2],ancilla_q[0]]
        X & clause_q[2]
        CCCX([0,1,3,clause_q[2],ancilla_q[1]]) | cgate
        for i in [0,1,3]:
            X & i
        # sentence4
        for i in [0,1,2,3]:
            X & i
        CCCX([0,1,2,ancilla_q[0],ancilla_q[1]]) | cgate
        X & ancilla_q[0]
        CCX & [3,ancilla_q[0],clause_q[3]]
        X & ancilla_q[0]
        CCCX([0,1,2,ancilla_q[0],ancilla_q[1]]) | cgate
        for i in [0,1,2,3]:
            X & i
    return 7, cgate

########################################################################

import cupy as cp

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

########################################################################

n = 8
k, oracle = puzzle_oracle()
circ = Grover(StateVectorSimulator()).circuit(n, k, oracle, measure=False)

amp = StateVectorSimulator().run(circ)
names = [bin(i)[2:].rjust(n,'0') for i in range(1<<n)]
values = trace_prob(amp,list(range(n))).reshape([1<<n])
print(f"success rate = {sum([values[i] if bin(i)[2:].rjust(n,'0')=='00101111' else 0 for i in range(len(values))])}")

import matplotlib.pyplot as plt
plt.figure(figsize=(30,6))
plt.bar(names, values)
plt.xticks(fontsize=6,rotation=75)
plt.yticks(ticks=[i/10 for i in range(11)])
plt.show()

# ```
#     2023-02-06 10:10:27 | circuit | INFO | Initial Quantum Circuit circuit_6c4d093aa5c311ed91250242ac110007 with 15 qubits.
#     2023-02-06 10:10:27 | Grover | INFO | 
#     circuit width          =   15
#     oracle  calls          =   12
#     other circuit size     =  872
    
#     success rate = 0.9999470421015482
# ```



# ![png](../../../assets/images/tutorials/algorithm/quantum_algorithm/tutorial_grover.nbconvert_11_1.png)
    


# 在同一个谕示电路上运行PartialGrover模块：


# ```python
n = 8
n_block = 3
print(f"run with n = {n}, block size = {n_block}")
k, oracle = puzzle_oracle()
circ = PartialGrover(StateVectorSimulator()).circuit(n, n_block, k, oracle, measure=False)

amp = StateVectorSimulator().run(circ)
names = [bin(i)[2:].rjust(n,'0') for i in range(1<<n)]
values = trace_prob(amp,list(range(n))).reshape([1<<n])
print(f"success rate = {sum([values[i] if bin(i)[2:].rjust(n,'0')[:3]=='001' else 0 for i in range(len(values))])}")

import matplotlib.pyplot as plt
plt.figure(figsize=(30,6))
plt.bar(names, values)
plt.xticks(fontsize=6,rotation=75)
plt.yticks(ticks=[i/10 for i in range(11)])
plt.show()
# ```
# ```
#     run with n = 8, block size = 3
#     2023-02-06 10:11:00 | circuit | INFO | Initial Quantum Circuit circuit_80687efea5c311ed91250242ac110007 with 16 qubits.
#     2023-02-06 10:11:01 | Grover-partial | INFO | 
#     circuit width           =   16
#     global Grover iteration =    8
#     local  Grover iteration =    3
#     oracle  calls           =   12
#     other circuit size      = 1048

#     success rate = 0.797729908327687
# ```




# ![png](../../../assets/images/tutorials/algorithm/quantum_algorithm/tutorial_grover.nbconvert_13_3.png)
    

