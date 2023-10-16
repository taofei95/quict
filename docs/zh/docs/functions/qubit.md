# 量子比特 (Qubit)

比特 (bit) 是经典计算和经典信息的基本信息量单位，量子计算也有着类似的结构，量子比特 (qubit) 。对于每一个经典比特都有一个状态，要么是 $0$ ，要么是 $1$ 。量子比特也同样有一个状态，量子比特的两个可能的状态是 $|0⟩$ 和 $|1⟩$ ，如你所想，它们分别对应于经典比特的状态 $0$ 和 $1$ 。

比特和量子比特之间的区别在于量子比特可以处于除 $|0⟩$ 或 $|1⟩$ 外的状态，量子比特可以是状态的线性组合，通常称为叠加态，如:

$$
|\psi \rangle \rightarrow \alpha |0 \rangle + \beta |1 \rangle
$$

其中 $\alpha$ 和 $\beta$ 是复数。换句话说，量子比特的状态 是二维复向量空间中的向量。特殊的 $|0⟩$ 和 $|1⟩$ 状态被称为计算基矢态，是构成该向量空间的一组正交基。

在 QuICT 中，我们使用 Qubit 类来表示量子计算中量子比特的概念。我们还构建了 Qureg 类来储存和管理一个 Qubit 类的列表， Qureg 类继承自 python 的 List 类，使其可以被当作 python 的 List 使用。

``` python
from QuICT.core import Qureg, Qubit

qubit = Qubit()
qubit.id        # 量子比特的唯一ID
qubit.measured  # 经过量子测量门之后的量子比特的状态

qr1 = Qureg(5)          # 构建 5 qubit 的 Qureg
qr2 = Qureg([qubit])    # 构建包含 qubit 的 Qureg
```

## 物理仿真参数
在真实量子机中，量子比特有一些关键参数，包括：
- 相干时间 T1
- 退相干时间 T2
- 比特测量保真度 F0/F1
- 态制备保真度
- 单/双比特门保真度
- 工作频率
- 测量频率

在QuICT 中设置相关量子比特参数 （可以在VQM中进行更全面的量子机仿真）
``` python
from QuICT.core.utils import GateType
qubit = Qubit()
qubit.fidelity = (0.995, 0.989)         # Set Measured Fidelity(F0, F1) for qubit
qubit.preparation_fidelity = 0.976      # Set preparation fidelity for qubit
qubit.gate_fidelity = {GateType.rx: 0.991, GateType.ry: 0.992, GateType.rz: 0.989}      # Set Gate fidelity, you can use simple float for average single-qubit Gate fidelity. qubit.gate_fidelity = 0.991
qubit.T1 = 4.68     # Set T1 time for qubit
qubit.work_frequency = 5.68     # Set working frequency for qubit   

# Set Coupling Strength for Qureg (bi-qubits Quantum Gate Fidelity)
qureg = Qureg(3)
coupling_strength = [(0, 1, 0.893), (1, 2, 0.993)]  # (start_qubit, end_qubit, fidelity)
qureg.set_coupling_strength(coupling_strength)
```
```python
# Qubit infomation
qubit id: ea3d71b19c754c8785662c5bbbf278f4; fidelity: (0.995, 0.989); QSP_fidelity: 0.976; Gate_fidelity: {<GateType.rx: 'Rx gate'>: 0.991, <GateType.ry: 'Ry gate'>: 0.992, <GateType.rz: 'Rz gate'>: 0.989}; Coherence time: T1: 4.68; T2: 0.0; Work Frequency: 5.68; Readout Frequency: 0.0; Gate Duration: 0.0
```


## 量子比特测量

我们可以通过检查一个比特来确定它处于 $0$ 态还是 $1$ 态。例如，计算机读取其内存内容时始终执行此操作。但值得注意的是，我们不能通过检查量子比特来确定它的量子态，即 $\alpha$ 和 $\beta$ 的值。相反，量子力学告诉我们，我们只能获得有关量子态的有限信息。

在测量量子比特时，我们以 $|\alpha|^2$ 的概率得到结果 $0$ ，以 $|\beta|^2$ 的概率得到结果 $1$ 。显然，$|\alpha|^2 + |\beta|^2 = 1$ ，因为概率和为 $1$ 。 从几何上看，我们可以将此解释为量子比特的状态归一化长度为 $1$ 。因此，通常量子比特的状态是二维复向量空间中的单位向量。

在QuICT中，我们可以通过在量子电路的对应比特位上放置测量门来实现量子比特的测量。

``` python
from QuICT.core import Circuit
from QuICT.core.gate import Measure
from QuICT.simulation.state_vector import StateVectorSimulator


circuit = Circuit(5)        # 构建 5-qubit 的量子电路
circuit.random_append(20)   # 在电路中随机放置 20 个量子门
Measure | circuit           # 将测量门放置在所有量子比特上

# 量子电路模拟
sim = StateVectorSimulator()
sv = sim.run(circuit)
print(int(circuit.qubits))  # 展示量子比特的最终测量结果
```


## 量子寄存器 (QuReg)
量子比特寄存器是用于存储和控制多个量子比特的集合，它的状态可以描述为多个量子比特的叠加态。在 QuICT 中我们使用 Qureg 类来实现量子比特寄存器的功能。

``` python
from QuICT.core import Qureg

qureg = Qureg(5)    # 5比特的量子比特寄存器

# Set Fidelity and T1 for Qureg
qureg.set_fidelity([0.5] * 5)
qureg.set_t1_time([30.1] * 5)
print(qureg[0])     # show the details about first qubit in current Qureg
```
``` python
qubit id: dfa14db83ac24925a6796f14d6874bba; fidelity: 0.5; QSP_fidelity: 1.0; Gate_fidelity: 1.0; Coherence time: T1: 30.1; T2: 0.0; Work Frequency: 0.0; Readout Frequency: 0.0; Gate Duration: 0.0
```
