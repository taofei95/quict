# 蒙特卡罗树搜索量子电路映射算法

本节介绍如何使用蒙特卡洛树搜索来解决量子电路映射问题[<sup>[1]</sup>](#refer1)。

## 算法原理

面对当前不满足拓扑约束的电路，有多种潜在方案来处理它使得它满足约束。蒙特卡罗树搜索算法将这些潜在的可行状态构造为一棵搜索树，通过动态地平衡树的扩展与利用，来尽可能挑选最优的状态。

### 状态定义与评估

当前的逻辑电路 $LC$，物理电路 $PC$，逻辑比特-物理比特的映射关系 $\tau$ （表示为一个排列），共同构成了当前的状态：

$$
s=(\tau, PC, LC)
$$

每插入一个 SWAP 门，会使得比特映射关系改变，还可能使得部分逻辑电路中的门可以被执行，因此他们从 LC 中删去，放入了 PC 中，状态从 s 改变为了 s^'。
每个 SWAP 门对应一个短期奖励，定义为 SWAP 前后逻辑电路中门数量的减少量：

$$
RWD(s^\prime,s) = \#LC(s^\prime ) - \#LC(s)
$$

每个状态对应一个长期奖励值，表示算法对该状态下最终执行完逻辑电路的全部门所需开销的估计值：

$$
VAL(s) = \max \left\{ SIM(s), \gamma \cdot [RWD(s,s^{\prime\prime} ) + VAL(s^{\prime\prime} )] \right\}
$$

这一估计值是随着算法的执行逐步更新的。每个状态的初始估计值定义为 $SIM(s)$，是基于模拟模块给出的分数。随着搜索的过程，每个子节点状态的估计值，以及当前状态转移到子节点状态的奖励，共同更新当前状态的估计值。同样地，当前节点的估计值更新后，也会反馈到祖先节点的状态上。更新值的反馈有一个预定义的常数衰减系数 $\gamma < 1$，控制算法只考虑较近的子孙节点状态的更新，近似忽略过于遥远的子孙节点状态更新。

### 选择

选择算法选择某个叶子节点，用于后续的扩展操作。选择算法尽可能平衡地利用已经搜索过的高估计值节点，和尚未充分探索过子树的节。算法从根节点开始，每次选择最大化下式的一个子节点，直到到达某个叶子节点，在选择过程中更新经过路径上节点的访问计数器。

$$
RWD(s, s^{\prime\prime}) + VAL(s^\prime) + c\sqrt{\frac{\log{\# VISIT(s)}}{\# VISIT(s^\prime)}}
$$

### 扩展

当前选择的叶子节点，在此基础上有若干不同方法插入一个 SWAP 门，使得该节点产生若干不同的子节点。为了优化搜索性能，这里限制插入的 SWAP 门至少有一个端点与该节点的逻辑电路 $LC$ 第一层量子门对应的比特 $Q_0$ 重合，即只允许选择交换 $(v_i,v_j )$ 的 SWAP 门：

$$
SWAP_{LC, \tau} = (v_i, v_j) \in E \land (\tau^{-1}(v_i) \in Q_0 \lor \tau^{-1}(v_j) \in Q_0)
$$

其中 τ 表示当前的比特映射。

### 模拟

上一步产生的新节点成为了搜索树中的新叶子节点，它们每个节点都需要通过模拟算法，给出长期奖励的估计值。出于性能的考虑，模拟算法不会把当前逻辑电路的全部门都用于模拟，而从中取出最多 $G_{SIM}$ 个门用于进行模拟。设 $N_{SIM}$ 轮随机模拟中，将这 $G_{SIM}$ 个门全部映射到物理设备上所需的 SWAP 门个数的最小值为 $N$，那么模拟算法的结果就是

$$
SIM(s) = \gamma ^ {\frac{N}{2}} \cdot G_{SIM}
$$

这里衰减系数的幂次 $\frac{N}{2}$ ，表示尽管这 $N$ 个门分布在当前节点后 $N$ 个节点的状态转移中，但平均地认为它们都是在 $\frac{N}{2}$ 位置一次性作用。这个处理主要是为了保证模拟结果本身与后续更新模拟结果时使用的奖励衰减含义吻合。

下面描述单轮随机模拟的过程。仍然按照扩展时的限制，只允许选择前 $G_{SIM}$ 个门构成的子电路 $C$ 中，与当前逻辑电路第一层有一端重合的 $h \in SWAP_{C, \tau}$。对每个可行的选择 $h$，通过它对逻辑电路第一层门 $L_0 (C)$ 的作用，定义它的影响因子：

$$
IF(h) = f \left( \sum_{g \in L_0(C)} SCOST(g, \tau) - \sum_{g \in L_0(C)} SCOST(g, \tau^\prime) \right)
$$

其中，$SCOST(g, \delta)$ 表示逻辑电路中的某个门 $g$ 在映射 $\delta$ 下，两端点在物理拓扑图上的距离。求和表示对当前逻辑电路中的第一层中所有门 $SCOST$ 求和。两个求和相减，表示了该 SWAP 操作多大程度上使得目前逻辑电路中无法执行的那些门靠近变得接近于可执行的状态。$f$ 是一个调整函数，用于舍弃负数取值（丢弃那些使得门的端点更远的 SWAP），它的定义类似于 $Relu$，只是在 $x=0$ 处添加了一个极小的偏移，避免算法过早丢弃那些看起来并不足够好的 $h$：

$$
f(x) = ReLU(x) + \epsilon [x=0]
$$

归一化影响因子，得到概率分布

$$
P(X = h) = \frac{IF(h)}{\sum_{h^\prime} IF(h^\prime)}
$$

基于这一概率分布，对所有可行解进行随机采样，采样结果用于状态转移，直到所有门都被执行。

### 反向传播

模拟结束后，反向更新该节点的所有祖先节点。假设 $s^\prime$ 是孩子节点， $s$ 是父亲节点，传播公式为

$$
VAL(s) \leftarrow \max\left\{ VAL(s), \gamma \cdot [RWD(s, s^\prime) + VAL(s^\prime)] \right\}
$$

该更新过程从新叶子一路更新回根节点。

### 决策

执行完 $N_{BP}$ 轮的选择、扩展、模拟、反向传播过程后，我们认为当前的搜索树已经被充分探索过，测试选择根节点 $rt$ 的最大化 $RWD(rt, s) + VAL(s)$ 的子节点 $s$，用 $s$ 作为新的根节点，执行后续的搜索算法。

## 示例代码

以下代码加载 IBMQX2 拓扑结构，随机生成一个包含 10 个门的电路后执行映射。

```python
import os

from QuICT.core import Circuit, Layout
from QuICT.core.utils import GateType
from QuICT.qcda.mapping import MCTSMapping


if __name__ == '__main__':
    layout_path = os.path.join(os.path.dirname(__file__), "../layout/ibmqx2_layout.json")
    layout = Layout.load_file(layout_path)

    circuit = Circuit(5)
    circuit.random_append(10, typelist=[GateType.cx])
    circuit.draw(filename='before_mapping')

    mcts = MCTSMapping(layout)
    circuit_map = mcts.execute(circuit)
    circuit_map.draw(filename='after_mapping')
```

## 参考文献

<div id="refer1"></div>

<font size=3>
[1] Zhou X, Feng Y, Li S. Quantum Circuit Transformation: A Monte Carlo Tree Search Framework. ACM Transactions on Design Automation of Electronic Systems (TODAES). 2022 Jun 27;27(6):1-27. (https://arxiv.org/abs/2008.09331)
</font>