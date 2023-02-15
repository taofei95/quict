# Grover搜索算法

## 概要

Grover搜索算法是一种无结构量子搜索算法，对于单个目标的情况，其查询复杂度是 $O(\sqrt{N})$ ，其中 $N$ 是状态空间大小。

这里实现了教科书版本[<sup>[1]</sup>](#refer1)的Grover算子，此外还支持多解情况，支持bit-flip和phase-flip oracle。当解的数目所占比例较大（超过一半）时，Grover迭代次数为0，算法退化为均匀随机采样。

Grover搜索算法的实际运行时间取决于谕示（oracle）电路的复杂程度。对于20个变量，解数目比例为 $2.2\times10^{-3}$ 的SAT问题，算法在单块GPU上可以在一小时内完成。

## 算法设置[<sup>[2]</sup>](#refer2)

量子计算机在无结构搜索问题上的二次加速可能是其相对于经典计算机最知名的优势之一。Grover的算法展示了这种能力，该算法不但可以二次加速非结构化搜索问题，也可以用作一般技巧或子例程，以获得各种其他算法的二次加速，这种技巧常被称作振幅放大（Amplitude Estimation）。

### 无结构搜索问题

考虑一个有 $N$ 个项目的列表。在这些项目中，有一个项目具有我们希望找到的独特属性，我们将这个称为赢家w。将列表中的每个项目视为特定颜色的框。假设列表中除获胜者外的所有项目均为灰色w，是紫色的。

<figure markdown>
![grover_list](../../../assets/images/tutorials/algorithm/quantum_algorithm/grover_list.png)
</figure>

为了找到紫色的盒子--*标记的项目*--使用经典计算，我们必须平均检查 $N/2$ 的这些盒子，在最坏的情况下，必须检查所有 $N$ 的盒子。然而，在量子计算机上，我们可以用Grover的振幅放大技巧在大约 $\sqrt{N}$ 的步骤中找到标记的项目。二次加速对于在长列表中寻找有标记的项目来说确实是一个可观的时间节省。此外，该算法不使用列表的内部结构，这使它成为*通用的；*这就是为什么它对许多经典问题立即提供了二次量子化的加速。

### 谕示电路的构造

对于本教科书中的例子，我们的“数据库”是由我们的量子比特可能处于的所有计算基础状态组成的。例如，如果我们有3个量子比特，我们的列表是 $|000\rangle, |001\rangle, \dots |111\rangle$ （即 $|0\rangle \rightarrow |7\rangle$ 的状态）。

Grover算法需要的谕示电路输入翻转了标记状态的相位。也就是说，对于计算基础中的任何状态 $|x\rangle$ ，有：

$$
U_\omega|x\rangle = \bigg\{
\begin{aligned}
\phantom{-}|x\rangle \quad \text{if} \; x \neq \omega \\
-|x\rangle \quad \text{if} \; x = \omega \\
\end{aligned}
$$

谕示电路是一个对角线矩阵，其中对应于被标记项目的条目有一个相位翻转。例如，如果我们有三个量子比特，$\omega = \text{101}$，我们的神谕将有这样的矩阵：

$$
U_\omega = 
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & -1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}
\begin{aligned}
\\
\\
\\
\\
\\
\\
\leftarrow \omega = \text{101}\\
\\
\\
\\
\end{aligned}
$$

Grover算法的强大之处在于，它很容易将一个问题转换为这种形式的谕示电路。有许多计算问题很难_找到_一个解决方案，但相对来说却很容易_验证_一个解决方案。例如，我们可以通过检查所有的规则是否被满足来轻松验证[数独](https://en.wikipedia.org/wiki/Sudoku)的解决方案。对于这些问题，我们可以创建一个函数 $f$ ，它接收一个解决方案 $x$ ，如果 $x$ 不是一个解决方案（ $x\neq\omega$ ），则返回 $f(x)=0$ ，如果是一个有效的解决方案（ $x=\omega$ ），则返回 $f(x)=1$ 。那么我们的神谕可以描述为

$$U_\omega|x\rangle = (-1)^{f(x)}|x\rangle$$

而神谕的矩阵将是一个对角线矩阵的形式。

$$
U_\omega = 
\begin{bmatrix}
(-1)^{f(0)} &   0         & \cdots &   0         \\
0           & (-1)^{f(1)} & \cdots &   0         \\
\vdots      &   0         & \ddots & \vdots      \\
0           &   0         & \cdots & (-1)^{f(2^n-1)} \\
\end{bmatrix}
$$

### Grover算子

考虑这样的一个电路：

$$\mathcal{G}=U_s U_f, \quad U_s = I-2|s⟩⟨s|s⟩, \quad U_f = I-2|\omega⟩⟨\omega|\omega⟩$$

其中 $|s⟩$ 是均匀叠加态而 $|\omega⟩$ 是标记状态。在 $|\omega⟩$ 与 $\frac{1}{N-1}\sum_{x\neq\omega}|x⟩$ 构成的平面上，这个电路将状态做了一个逆时针旋转。

<figure markdown>
![grover_step3](../../../assets/images/tutorials/algorithm/quantum_algorithm/grover_step3.jpg)
</figure>

两个反射总是对应于一个旋转。Grover算子使初始状态 $|s\rangle$ 向标记状态 $|w\rangle$ 旋转。振幅条形图中的反射 $U_s$ 的作用可以理解为对平均振幅的反射，而 $U_f$ 则是对非标记状态的反射。这个过程将重复数次，以锁定标记状态。经过 $t$ 步，我们将处于 $|\psi_t\rangle$ 状态，其中 $| \psi_t \rangle = (U_s U_f)^t | s \rangle$ 。事实证明，大约 $\sqrt{N}$ 的旋转就足够了。这一点在观察状态 $| \psi \rangle$ 的振幅时就很清楚了。我们可以看到，$| w\rangle$ 的振幅随着应用次数 $\sim t N^{-1/2}$ 线性增长。然而，由于我们处理的是振幅而不是概率，矢量空间的维数以平方根的形式进入。因此，在这个过程中，被放大的是振幅，而不仅仅是概率。

在有多个解决方案的情况下，可以证明大约 $\sqrt{(N/M)}$ 的旋转就足够了，其中 $M$ 是解的数目。


## 基本用法

`Grover`类位于`QuICT.algorithm.quantum_algorithm.grover`，`circuit`/`run`的参数包括：

1. `n`：oracle状态空间向量的位数
1. `n_ancilla`：oracle辅助比特的位数
1. `oracle`：所使用的oracle电路
1. `n_solution`：解的数量，解数目未知时传入`None`。默认为`1`
1. `measure`：最终的电路是否包含测量。默认为`True`

用户的oracle可以自行构建，也可以使用框架中已有的oracle（MCT oracle、CNF oracle）。随后用户实例化一个`Grover`对象，然后调用`circuit`方法得到电路或者调用`run`方法运行Grover搜索算法得到搜索结果。

## 代码示例

### 单个解的搜索

在4位的MCT oracle上执行搜索。

```python
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.core.gate.backend import MCTOneAux

def main_oracle(n, f):
    result_q = [n]
    cgate = CompositeGate()
    target_binary = bin(f[0])[2:].rjust(n, "0")
    with cgate:
        X & result_q[0]
        H & result_q[0]
        for i in range(n):
            if target_binary[i] == "0":
                X & i
    MCTOneAux().execute(n + 2) | cgate
    with cgate:
        for i in range(n):
            if target_binary[i] == "0":
                X & i
        H & result_q[0]
        X & result_q[0]
    return 2, cgate

n = 4
target = 0b0110
f = [target]
k, oracle = main_oracle(n, f)
grover = Grover(simulator=StateVectorSimulator())
result = grover.run(n, k, oracle)
print(result)
```

更多的例子见`example/demo/tutorial_grover.ipynb`和`unit_test/algorithm/quantum_algorithm/grover_unit_test.py`。

## 参考文献

<div id="refer1"></div>

<font size=3>
[1] Nielsen, M. A., & Chuang, I. L. (2019). *Quantum computation and quantum information*. Cambridge Cambridge University Press. [doi:10.1017/CBO9780511976667](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview)
</font>

<div id="refer2"></div>

<font size=3>
[2] Grover’s Algorithm. (n.d.). Community.qiskit.org. https://qiskit.org/textbook/ch-algorithms/grover.html
</font>