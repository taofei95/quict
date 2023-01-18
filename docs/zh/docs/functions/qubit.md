# 量子比特 (Qubit)

比特(bit)是经典计算和经典信息的基本信息量单位，量子计算也有着类似的结构，量子比特(qubit)。对于每一个经典比特都有一个状态，要么是 $0$，要么是 $1$。量子比特也同样有一个状态，量子比特的两个可能的状态是 $|0⟩$ 和 $|1⟩$，如你所想，它们分别对应于经典比特的状态 $0$ 和 $1$。

比特和量子比特之间的区别在于量子比特可以处于除 $|0⟩$ 或 $|1⟩$ 外的状态，量子比特可以是状态的线性组合，通常称为叠加态，如:

$$
|\psi \rangle \rightarrow \alpha |0 \rangle + \beta |1 \rangle
$$

其中 $\alpha$ 和 $\beta$ 是复数。换句话说，量子比特的状态 是二维复向量空间中的向量。特殊的 $|0⟩$ 和 $|1⟩$ 状态被称为计算基矢态，是构成该向量空间的一组正交基。

在QuICT中，我们使用 Qubit 类来表示量子计算中量子比特的概念。我们还构建了 Qureg 类来储存和管理一个Qubit类的列表，Qureg类继承自python的List类，使其可以被当作 python 的 List 使用。

```python
from QuICT.core import Qureg, Qubit

qubit = Qubit()
qubit.id        # 量子比特的唯一ID
qubit.measured  # 经过量子测量门之后的量子比特的状态

qr1 = Qureg(5)          # 构建 5 qubit 的 Qureg
qr2 = Qureg([qubit])    # 构建包含 qubit 的 Qureg
```

## 量子比特测量
----
我们可以通过检查一个比特来确定它处于 $0$ 态还是 $1$ 态。例如，计算机读取其内存内容时始终执行此操作。但值得注意的是，我们不能通过检查量子比特来确定它的量子态，即 $\alpha$ 和 $\beta$ 的值。相反，量子力学告诉我们，我们只能获得有关量子态的有限信息。

在测量量子比特时，我们以 $|\alpha|^2$ 的概率得到结果$0$，以 $|\beta|^2$ 的概率得到结果$1$。显然，$|\alpha|^2 + |\beta|^2 = 1$，因为概率和为 $1$。 从几何上看，我们可以将此解释为量子比特的状态归一化长度为 $1$。因此，通常量子比特的状态是二维复向量空间中的单位向量。

> 在QuICT中，我们可以通过在量子电路的对应比特位上放置测量门来实现量子比特的测量。
```python
from QuICT.core import Circuit
from QuICT.core.gate import Measure
from QuICT.simulation.state_vector import CircuitSimulator


circuit = Circuit(5)        # 构建 5-qubit 的量子电路
circuit.random_append(20)   # 在电路中随机放置 20 个量子门
Measure | circuit           # 将测量门放置在所有量子比特上

# 量子电路模拟
sim = CircuitSimulator()
sv = sim.run(circuit)
print(int(circuit.qubits))  # 展示所有比特的测量结果
```