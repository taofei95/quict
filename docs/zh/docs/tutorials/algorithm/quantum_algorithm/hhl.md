

# HHL求解线性方程算法 



HHL算法[^1]是一个用于解决线性系统问题的算法，比经典算法具有指数加速，是许多重要量子计算算法的基础。本教程旨在介绍如何使用QuICT的HHL模块，并结合代码实际实例进一步阐述此算法。

## 算法原理

求解线性方程问题$\mathbf{A}\vec{x}=\vec{b}$，在经典算法上大多都是求解$\mathbf{A}$的逆$\mathbf{A}^{-1}$再作用于$\vec{b}$上以获得解$\vec{x}=\mathbf{A}^{-1}\vec{b}$。考虑到可逆矩阵$\mathbf{A}$与他的逆$\mathbf{A}^{-1}$具有相同的特征向量，且齐对应的特征值互为倒数，所以通常的解法都是将矩阵$\mathbf{A}$进行对角化，再将对角元变成倒数后再逆对角化，最后乘以向量$\vec{b}$得到解。

###  量子相位估计

假设我们通过量子初态制备(Quantum State Preparation)或者其他的量子算法得到了一个初态$\ket b$，我们希望求解出$\ket x = \mathbf A^{-1}\ket b$。在量子系统中，我们把算符$\mathbf A$写作$\sum \lambda_j\ket{u_j}\bra{u_j}$，其中$\lambda_j$表示$\mathbf A$的特征值，$\ket{u_j}$为对应的特征向量；初态$\ket b$写作$\sum b_j\ket{u_j}$，表示$\ket b$在$\mathbf A$上投影的态叠加。那么目标解$\ket x=\sum \lambda_j^{-1}b_j\ket{u_j}$为算法需要输出的状态。

我们知道在量子计算中，量子相位估计(Quantum Phase Estimation)可以估计出酉算子$U$在态$\ket{u}$下对应的特征值$e^{2\pi i\varphi}$的相位$\varphi$。具体细节可以在《QCQI》[^2]中5.2节看到，在此不加赘述。我们需要讨论的是这一部分如何作用在线性方程中。

量子相位估计要求矩阵$U$是酉的，那么我们可以将Hermitian$\mathbf A$通过自然指数构造成一个酉矩阵$U=e^{iAt}$。在量子系统中，它表示为$U=\sum e^{i\lambda_jt}\ket{u_j}\bra{u_j}$，那么使用QPE，我们可以将$\lambda_j$的估计值作为状态$\ket{\tilde{\lambda_j}}$提取到相位寄存器，通过$t$位无符号二进制表示，其中$t$是相位寄存器的比特数。需要注意的是， 在量子模拟中我们不考虑时间的作用，故通常取$t$为常数$2\pi$，并且假设$\forall \lambda_j\in(0,1]$。

经过部分电路后，整个系统的状态为
$$
\sum \ket{\tilde{\lambda_j}}\cdot b_j\ket{u_j}
$$

### Rotation

经过QPE，我们获得了在相位寄存器上的含有$\mathbf A$的估计特征值的状态$\ket{\tilde{\lambda_j}}$，接下来，我们需要让这些特征值$\lambda_j$取倒数即$\lambda_j^{-1}$作为它们对应特征向量$\ket{u_j}$的振幅。

我们引入一个新的辅助比特，它的初态为，这样整个系统的状态为$\sum\ket0\cdot \ket{\tilde{\lambda_j}}\cdot b_j\ket{u_j}$。我们很容易想到使用受控旋转门，将相位寄存器所有比特作为控制位，按照它的值$\tilde{\lambda_j}$进行旋转，于是在这部分结束后获得状态。
$$
\sum\Big(\sqrt{1-\frac{C^2}{\tilde{\lambda_j}^2}}\ket 0 + \frac{C}{\tilde{\lambda_j}}\ket 1\Big)\cdot\ket{\tilde{\lambda_j}}\cdot b_j\ket{u_j}
$$
相当于在辅助比特上进行多控Ry操作，其中参数为$2\arcsin(\frac C {\tilde{\lambda_j}})$，$C$为常数。我们需要的是辅助比特在$\ket 1$上的状态，所以需要对其进行测量。其中测量结果为$\ket 1$的概率为
$$
p_1=\sum(\frac {Cb_j}{\tilde{\lambda_j}})^2=C^2\sum(\frac {b_j}{\tilde{\lambda_j}})^2
$$
即$C$取值需要尽可能的大，且满足$\forall \frac{C}{\tilde{\lambda_j}}\le1$，即$C\le \tilde\lambda_{min}$。



最后，作用inverseQPE，将$\ket{\tilde{\lambda_j}}$还原成$\ket 0^{\otimes t}$，输入比特便从算法开始时的初态$\ket b=\sum b_j\ket{u_j}$变成了$\ket {\hat x}=\sum \lambda^{-1}b_j\ket{u_j}$。

### 其他情况

我们在上面给出了HHL算法输入的假设

* 所有输入是规范的，即$\mathbf A$为$2^N\times2^N$的方阵，并且可逆

* 求解的线性方程矩阵$\mathbf A$需要时厄米共轭的（即$\mathbf A = \mathbf A^\dagger$）

* $\mathbf A$的所有特征值都需要在区间$(0,1]$上

假设一量子系统中必须满足的条件，所以应该严格遵守。

我们先考虑假设二不成立，此时自然指数$e^{iAt}$非酉，不满足量子算符的条件。然而，我们可以通过构造一组新的输入，来保证满足厄米共轭的条件。
$$
A'=
\begin{bmatrix}
0 & A^\dagger\\
A & 0
\end{bmatrix}
$$
此时，在初态$\ket b$上添加一个状态为$\ket 1$的比特，于是输入寄存器的状态为
$$
\ket {b'}=\ket 1\ket b=
\begin{bmatrix}
0\\
b
\end{bmatrix}
$$
那么在算法成功后，输入寄存器获得的状态为
$$
\ket{x'}=\ket0\ket x=
\begin{bmatrix}
x\\
0
\end{bmatrix}
$$
包含我们需要的状态x。



接下来考虑假设三不成立，如果$\mathbf A$的特征值大于1，只需要将$\mathbf A$乘上一个规范系数$k$，使得$k\mathbf A$的特征值在$(0,1]$上即可。

根据假设二的构造，我们获得的厄米矩阵$A'$的特征值为$\{\lambda_j,-\lambda_j|\lambda_j为A的特征值\}$，由于$A$可逆，此时必然同时包含两种符号的特征值。当然，这只是特例，其他厄米矩阵$A$同样可能包含两种符号的特征值。

在《QCQI》[^2]中量子相位估计这一节，使用的是$e^{2\pi i\varphi}$这个周期为1的周期函数，定义域为$(0,1]$。如果我们将定义域变至$(-0.5,0.5]$，这个函数周期依然为1，但是自变量可以为负数了。为了方便描述，我们将函数改写成$e^{\pi i\varphi}，\varphi\in(-1,1]$。

根据这一想法，我们将相位寄存器的其中一个量子比特作为二进制符号位，在相位寄存器中按照一个符号位的二进制编码存储$\mathbf A$的特征值状态$\ket{\tilde{\lambda_j}}$，同样对Rotation部分稍作修改，我们就获得了可以处理带符号特征值的$\mathbf A$的电路。

## 用QuICT实现HHL算法

`HHL`类位于`QuICT.algorithm.quantum_algorithm.hhl`，`circuit`的参数包括：

1. `matrix`：需要求解的矩阵$\mathbf A$
2. `vector`：需要求解的向量$\vec b$
3. `dominant_eig`：$\mathbf A$的主特征值估计值，即$\max|\tilde\lambda|$。若空则使用`numpy.linalg.eigvals`求出
4. `min_abs_eig `：$\mathbf A$的最小特征值估计值，即$\min|\tilde\lambda|$。若空则使用`numpy.linalg.eigvals`求出
5. `phase_qubits`：在相位估计中所使用的相位寄存器的量子比特数，与解的精度有关。默认值为9
6. `control_unitary`：在电路中实现CU门的方法
7. `measure`：是否需要对辅助比特进行测量。默认为True

在使用`hhl.circuit()`成功创建电路后，接着使用`hhl.run()`来运行该电路。如果需要重置电路，请使用`hhl.reset()`

在该算法中，为了保证不丢失解的精度，Rotation部分的C取值足够小，这导致算法中辅助比特测量为$\ket1$的成功率较低，于是可以让用户选择是否需要measure操作来去除辅助位。若删除measure门，则目标状态需要在辅助比特为$\ket 1$时进行后续操作。

用户需要实例化一个`HHL`对象，随后调用`circuit`方法获得电路，然后调用`run`来使用HHL算法求得线性方程近似解。

### 代码实例

我们对于给出的非厄米矩阵$\mathbf A$和向量$\vec b$，在相位寄存器量子比特数`phase_qubits`为8时，运行含measure门的电路，输出最后返回的解向量，并与`numpy.linalg.solve`的结果比较，同时计算算法的成功率。

```python
import numpy as np

from QuICT.algorithm.quantum_algorithm.hhl import HHL
from QuICT.simulation.state_vector import StateVectorSimulator


def MSE(x, y):
    n = len(x)
    res0 = np.dot(x + y, x + y) / n
    res1 = np.dot(x - y, x - y) / n
    return min(res0, res1)

A = np.array([[1.0 + 0j, 2.0 + 0j],
              [3.0 + 0j, 2.0 + 0j]])
b = np.array([1.0 + 0j, -2.0 + 0j])

slt = np.linalg.solve(A, b)
slt /= np.linalg.norm(slt)

hhl = HHL(StateVectorSimulator(device="GPU"))
hhl.circuit(
        A, b, phase_qubits=8
    )
time = 0
hhl_a = None
while hhl_a is None:
    hhl_a = hhl.run()
    time += 1


print(f"classical solution  = {slt}\n"
    + f"                hhl = {hhl_a}\n"
    + f"                MSE = {MSE(slt, hhl_a)}\n"
    + f"       success rate = {1.0 / time}"
    )
```

~~~
2023-08-07 17:36:18 | hhl | INFO | circuit width    =   11
circuit size     = 46944
hamiltonian size =  344
CRy size         = 46165
eigenvalue bits  =    8
classical solution  = [-0.76822128+0.j  0.6401844 +0.j]
                hhl = [-0.75919692-5.02380793e-15j  0.62410119+7.80407308e-15j]
                MSE = (0.00017005435770275006-1.7085119493572715e-16j)
       success rate = 1.0
~~~

可以看到在该输入下，电路只执行一次便成功，且返回的误差较小。

## 参考文献

[^1]:Harrow, A. W., & Hassidim, A., & Lloyd, S. (2009). *Quantum algorithm for linear systems of equations*. Physical review letters, 15 103, 150502.[arXiv:quant-ph/0811.3171](https://arxiv.org/abs/0811.3171)
[^2]:Nielsen, M. A., & Chuang, I. L. (2019). *Quantum computation and quantum information*. Cambridge Cambridge University Press. 217-225.[doi:10.1017/CBO9780511976667](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview)

 

---