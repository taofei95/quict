# 量子模拟

## 经典计算机上的量子计算模拟

如何验证量子算法的正确性是量子计算最重要的部分之一，使用经典机器模拟量子电路是验证量子算法的一种方式。在 QuICT 中，Simulator 用于模拟量子电路运行过程中量子比特的状态，目前支持三种量子电路模拟器，分别是酉矩阵、状态向量和密度矩阵模拟。

|  模拟器后端   |   CPU   |   GPU   |    multi-GPU   |
| ------      | ------- |  ------  |    ------    |
|   状态向量    |   &#10004;   |  &#10004;   |    &#10004;   |
|    酉矩阵     |   &#10004;   |  &#10004;   |    &#10008;   |
|   密度矩阵    |   &#10004;   |  &#10004;   |    &#10008;   |

模拟器将返回一个 Dict 数据结构，该数据结构存储有关量子电路模拟的信息

- ID：电路名称
- SHOTS：模拟的重复次数
  
- 模拟器参数：
    - 设备：硬件类型 [CPU, GPU]
    - 后端：模拟器的模式
    - 选项：模拟器的参数
  
- 数据：
    - 计数：每次模拟的测量结果
    - 状态向量：经过模拟之后的量子比特状态向量
    - 密度矩阵：经过模拟之后的量子比特密度矩阵 [只针对后端为密度矩阵模拟器]

``` python
from QuICT.core import Circuit
from QuICT.simulation import Simulator

# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(5)
circuit.random_append(rand_size=100)

# Initial Simulator
simulator = Simulator(
    device="CPU",
    backend="state_vector",
    precision="double"
)
result = simulator.run(circuit, shots=1000)    # get simulation's result
```

## 状态向量模拟器

状态向量表示量子比特在状态空间中的向量。量子比特是一个二维的状态空间。假设 $|0⟩$ 和 $|1⟩$ 形成了这个状态空间的一组标准正交基。那么这个状态空间的任意向量都可以写成

$$
|\psi⟩ = \alpha|0⟩ + \beta|1⟩
$$

其中 $\alpha$ 和 $\beta$ 是任意的复数。

对于多个量子比特的复合状态空间则是其独立量子比特状态空间的张量积，设多个量子比特编号为1到n，且量子比特 $i$ 对应的状态向量为 $|\psi_i⟩$，则整个多量子比特状态向量为

$$
|\psi_1⟩ \otimes |\psi_2⟩ \otimes \dots \otimes |\psi_n⟩
$$

当然量子比特的状态在电路中并不是一成不变的，状态向量的演化可用 酉变换(unitary transformation) 来描述。也就是说，量子比特在 $t_1$ 时所处的状态 $|\psi⟩$ 和在 $t_2$ 时所处的状态 $|\psi^′⟩$ 是通过一个仅与时间 $t_1$ 和 $t_2$ 有关的酉算子 $U$，即量子门联系起来的。

$$
|\psi^′⟩ = U|ψ⟩
$$

状态向量模拟器就是在给定的量子电路和初始量子比特状态下，不断演化量子比特状态的变化，直到量子电路结束并返回最终的量子比特状态。

```python
from QuICT.core import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator

# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(5)
circuit.random_append(rand_size=100)

# Simulate Quantum Circuit
simulator = StateVectorSimulator()
result = simulator.run(circuit)
```

## 酉矩阵模拟 (Unitary Simulator)

与状态向量模拟器不同的是，酉矩阵模拟器会先将量子电路内的所有量子门的矩阵，融合成为一个酉矩阵。之后，再将量子状态与酉矩阵相乘，生成最终的量子比特状态。

``` python
from QuICT.core import Circuit
from QuICT.simulation.unitary import UnitarySimulator

# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(5)
circuit.random_append(rand_size=100)

# Simulate Quantum Circuit
simulator = UnitarySimulator(device="CPU", precision="double")
result = simulator.run(circuit)
```

## 密度矩阵模拟器

与状态向量等价，我们也可以使用密度矩阵来描述量子系统的演化。假设一个量子系统以概率 $p_i$ 处于多个状态 $|\psi_i⟩$ 之一，其中 $i$ 是一个指标，我们将把 ${p_i, |\psi_i⟩}$ 称为一个纯态系综 (ensemble of pure states)。系统的密度矩阵定义为

$$
ρ \equiv \sum_i p_i|\psi_i⟩⟨\psi_i|
$$

密度矩阵的演化也是由酉算子 $U$ 描述的，如果系统初态为 $|\psi_i⟩$ 的概率为 $p_i$，那 么在演化之后，系统将以概率 $p_i$ 处于状态 $U|\psi_i⟩$。因此，密度算子的演化由下式描述:

$$
ρ \equiv \sum_i p_i|\psi_i⟩⟨\psi_i| \stackrel{U}{\longrightarrow} \sum_i p_i U|\psi_i⟩⟨\psi_i|U^† = UρU^†
$$

与状态向量等价，我们也可以使用密度矩阵来描述量子系统的演化。假设一个量子系统以概率 $p_i$ 处于多个状态 $|\psi_i⟩$ 之一，其中 $i$ 是一个指标，我们将把 ${p_i, |\psi_i⟩}$ 称为一个纯态系综 (ensemble of pure states)。系统的密度矩阵定义为
$ρ \equiv \sum_i p_i|\psi_i⟩⟨\psi_i|$
密度矩阵的演化也是由酉算子 $U$ 描述的，如果系统初态为 $|\psi_i⟩$ 的概率为 $p_i$，那 么在演化之后，系统将以概率 $p_i$ 处于状态 $U|\psi_i⟩$。因此，密度算子的演化由下式描述:
$ρ \equiv \sum_i p_i|\psi_i⟩⟨\psi_i| \stackrel{U}{\longrightarrow} \sum_i p_i U|\psi_i⟩⟨\psi_i|U^† = UρU^†$

``` python
from QuICT.core import Circuit
from QuICT.simulation.density_matrix import DensityMatrixSimulator

# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(5)
circuit.random_append(rand_size=100)

# Simulate Quantum Circuit
simulator = DensityMatrixSimulator(device="CPU", precision="double")
result = simulator.run(circuit)
```

## 高性能状态向量模拟器 （QuICT_sim)
    !!! warning
        使用前需要安装 QuICT_sim， 可以使用 ‘pip install quict-sim’ 来安装。

为达到更快的模拟速度，充分利用经典计算机的计算性能，QuICT 基于CPU的AVX指令集和SSE指令集开发了一款更高性能的状态向量模拟器。与QuICT的基础状态向量模拟器相比，高性能状态向量模拟器可达到数倍到数十倍的性能提升，并且随着比特数的增加，其优势也愈发明显。

# TODO： add graph between based and high-performance

### 如何使用高性能状态向量模拟器
``` python
from QuICT.core import Circuit
from QuICT.simulation.state_vector import HPStateVecotrSimulator

# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(5)
circuit.random_append(rand_size=100)

# Simulate Quantum Circuit
simulator = HPStateVecotrSimulator(precision="double")
result = simulator.run(circuit)
```

