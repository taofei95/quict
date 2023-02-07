# 量子游走算法

量子游走（Quantum Walk）有两种常见的模型，分别是带硬币的量子游走（Coined quantum walk）和Szegedy量子游走（Szegedy quantum walk），前者是在图的顶点上进行的游走，而后者是沿图的边进行的游走，它们在某些情况下是等价的。本教程旨在介绍如何使用QuICT中内置的量子游走模块，并结合量子游走搜索这一重要应用进一步阐述此算法。

!!! note

    本教程目前只针对**离散时间**的，**带硬币**的量子游走算法。

## 带硬币的量子游走算法

### 算法原理

带硬币的量子游走是在图的顶点上进行的游走，因此每个节点都可以被视为一个状态，粒子可以在有边相连的两个状态之间移动。在硬币模型中，包含两个量子态和两个算子：

- 两个量子态

    - 位置状态 $|\psi_t⟩$ ：用于描述粒子在 $t$ 时刻时的位置，即粒子在游走的过程中处于各个位置的叠加态。
    - 硬币状态 $|\psi_C⟩$ ：用于描述粒子下一步如何移动。

- 两个算子

    - 硬币算子 $C$ （Coin Operator）：一个酉矩阵，模拟抛硬币的过程以获得硬币状态 $|\psi_C⟩$。
    - 移动算子 $S$ （Shift Operator）：一个置换矩阵，根据硬币状态 $|\psi_C⟩$ 决定粒子的移动情况。

!!! note

    一种常用的硬币是Hadamard硬币：

    $$
    H=\frac{1}{\sqrt{2}} \begin{bmatrix}
    1 & 1\\
    1 & -1
    \end{bmatrix}
    $$

    Hadamard硬币可以视为公平硬币，能够使粒子处于相等的叠加状态。

带硬币的量子游走本质上就是，先抛硬币，然后根据抛硬币结果让粒子进行移动：

$$|\psi_{t}⟩ \overset{C}{\rightarrow} \overset{S}{\rightarrow} |\psi_{t+1}⟩$$

因此，单步的量子游走算子 $U$ 可以被表示为：

$$U = S \cdot C$$

### 算法流程

1. 设初态 $|\psi_0⟩=|0⟩$ ，运行 $t$ 轮量子游走，每轮迭代的步骤为：
   
      1. 执行抛硬币操作 $C$ 以获得 $|\psi_C⟩$ ，即对所有硬币寄存器应用相同的硬币 $c$ 。
      2. 根据抛硬币结果 $|\psi_C⟩$ 执行移动操作 $S$ 。
   
    即：
    
    $$|\psi_{t}⟩ = U^t|\psi_0⟩$$
   
2. 对最终的状态进行量子测量

### 用QuICT实现量子游走

#### 基本用法

`QuantumWalk` 位于 `QuICT.algorithm.quantum_algorithm.quantum_walk`，运行函数的输入参数为：

- `step`: 自定义的量子游走轮数
- `position`: 给定图的节点数
- `edges`:` 给定图的边
- `operators`: 定义对各节点的操作
- `coin_operator`: 定义抛硬币操作（酉矩阵）
- `switched_time`: 单次搜索的硬币操作次数
- `shots`: 采样次数

最终输出对最终状态测量后的采样结果。

#### 代码实例

接下来，我们将以下图为例使用QuICT进行量子游走模拟，初始位置为节点0。

<figure markdown>
![quantum_walk](../../../assets/images/tutorials/algorithm/quantum_algorithm/quantum_walk.png){:width="400px"}
</figure>

首先，导入运行库：

``` python
from QuICT.algorithm.quantum_algorithm import QuantumWalk
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.tools.drawer.graph_drawer import *
```

初始化状态向量模拟器和`QuantumWalk`模块：

``` python
simulator = StateVectorSimulator()
qw = QuantumWalk(simulator)
```

根据图定义节点数和边：

!!! warning "edge需要满足的条件"

    - edge列表每一项对应一个节点，因此edge列表长度应等于节点数。
    - 子列表代表与其对应节点相连的节点编号。
    - 所有子列表长度应相等，代表图的维度。不足时应使用重复节点补齐。
    - 子列表内部的元素是有顺序的。如此例中：节点0的第一维连接节点1，节点1的第二维连接节点0。

``` python
position = 4
edges = [[1, 3], [2, 0], [3, 1], [0, 2]]
```

用均匀的Hadamard硬币模拟1step的量子游走，并使用内置的画图函数画出1step后的采样结果：

``` python
sample = qw.run(step=1, position=position, edges=edges, coin_operator=H.matrix)
draw_samples_with_auxiliary(sample, 2, 1)
```

<figure markdown>
![QW_result](../../../assets/images/tutorials/algorithm/quantum_algorithm/QW_result.png){:width="500px"}
</figure>

可以见得1step后，原本处于节点0的量子游走到节点1和3上。

## 基于硬币的量子游走搜索算法

### 算法原理

### 算法流程

1. 初始化硬币寄存器和节点寄存器为所有状态的等权重叠加态，即在全部量子比特上施加 $H$ 门。
2. 对于总节点数为 $n$ ，标记节点数为 $m$ 的超立方体，运行 $O(1/\sqrt{\frac{m}{n}})$ 轮量子游走搜索，每轮迭代的步骤为：
   
      1. 对于给定的硬币谕示（Coin Oracle），对未标记节点的对应状态应用硬币 $C_0$ ，对标记节点的对应状态应用硬币 $C_1$ 。
      2. 执行移动操作（Shift Operator）。
   
3. 对最终的状态进行量子测量

### 用QuICT实现N维超立方体上的量子游走搜索

#### 基本用法

`QuantumWalkSearch` 位于 `QuICT.algorithm.quantum_algorithm.quantum_walk`，运行函数的输入参数为：

- `index_qubits`: 超立方体维度
- `targets`: 标记的节点编号
- `step`: 自定义的搜索轮数
- `r`: 硬币参数 $r$
- `a_r`: 硬币参数 $a_r$
- `a_nr`: 硬币参数 $a_{nr}$
- `switched_time`: 单次搜索的硬币操作次数
- `shots`: 采样次数

最终输出对最终状态测量后的采样结果。

#### 代码实例

接下来，我们将以5-cube为例使用QuICT进行量子游走搜索，即节点数为32个，标记节点4。

``` python
# 导入运行库
from QuICT.algorithm.quantum_algorithm import QuantumWalkSearch
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.tools.drawer.graph_drawer import *

# 初始化状态向量模拟器和QuantumWalkSearch模块：
simulator = StateVectorSimulator()
qws = QuantumWalkSearch(simulator)

# 开始搜索
N = 5
sample = qws.run(index_qubits=N, targets=[4], a_r=5 / 8, a_nr=1 / 8)

# 画出采样图
draw_samples_with_auxiliary(sample, N, int(np.ceil(np.log2(N))))
```

<figure markdown>
![QWS_result1](../../../assets/images/tutorials/algorithm/quantum_algorithm/QWS_result1.png){:width="500px"}
</figure>

!!! warning "硬币参数 $a_r$ 和 $a_{nr}$ 的选择"

    选择适合的硬币参数 $a_r$ 和 $a_{nr}$ 非常重要。特别地，当 $a_r$ 和 $a_{nr}$ 非常接近时，标记硬币将失去标记作用：

    ``` python
    sample = qws.run(index_qubits=N, targets=[4], a_r=1 / 8, a_nr=0.9 / 8)
    draw_samples_with_auxiliary(sample, N, int(np.ceil(np.log2(N))))
    ```

    <figure markdown>
    ![QWS_result3](../../../assets/images/tutorials/algorithm/quantum_algorithm/QWS_result3.png){:width="400px"}
    </figure>

特别地，QuICT支持多节点的量子游走搜索，如同时标记节点4和节点15：

``` python
sample = qws.run(index_qubits=N, targets=[4, 15], a_r=5 / 8, a_nr=1 / 8)
draw_samples_with_auxiliary(sample, N, int(np.ceil(np.log2(N))))
```

<figure markdown>
![QWS_result2](../../../assets/images/tutorials/algorithm/quantum_algorithm/QWS_result2.png){:width="500px"}
</figure>

但是搜索效果通常会随着标记节点的增加而下降。


## 参考文献

<div id="refer1"></div>

<font size=3>
[1] Portugal, Renato. Quantum walks and search algorithms. Vol. 19. New York: Springer, 2013. [DOI: https://doi.org/10.1007/978-3-319-97813-0](https://doi.org/10.1007/978-3-319-97813-0)
</font>

<div id="refer2"></div>

<font size=3>
[2] Neil Shenvi, Julia Kempe, and K. Birgitta Whaley. A Quantum Random Walk Search Algorithm. Phys. Rev. A. [arXiv:quant-ph/0210064 (2003)](https://arxiv.org/abs/quant-ph/0210064)
</font>

<div id="refer3"></div>

<font size=3>
[3] Hristo Tonchev. Alternative Coins for Quantum Random Walk Search Optimized for a Hypercube. Journal of Quantum Information Science, Vol.5 No.1, 2015. [DOI: 10.4236/jqis.2015.51002](https://www.scirp.org/journal/paperinformation.aspx?paperid=55017)
</font>
