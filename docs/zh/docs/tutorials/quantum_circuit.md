# 量子电路基础使用

量子电路 (Circuit) 是 QuICT 中最为基础的模块，无论是量子算法，还是优化映射，都需要构建相对应的量子电路。本文将介绍如何使用 QuICT 构建我们所需要的量子电路，以及如何通过量子计算模拟器得到相对应的结果。

<figure markdown>
![circuit_relationship](../assets/images/tutorials/circuit_relationship.png){:width="350px"}
</figure>


## 构建量子电路

构建量子电路是整个量子计算中最为重要的一部分，量子电路是由量子比特和量子门组成的，通过将量子比特和量子门进行特定组合可以实现所需要的功能。QuICT 提供了灵活便捷的构建电路方式，也内置了很多关于量子电路信息的功能，例如：深度、宽度、门数、矩阵等。另一方面，QuICT 也支持量子电路可视化，OPENQASM 转换器，动态构建量子电路多种辅助构建功能。

``` python
from QuICT.core import Circuit
from QuICT.core.gate import H, CX

# 构建 5 比特GHZ态量子电路
circuit = Circuit(5)
H | circuit(0)
CX | circuit([0, 1])
CX | circuit([1, 2])
CX | circuit([2, 3])
CX | circuit([3, 4])
```

### 组合量子门

组合量子门也可以被加入到量子电路中，作为量子电路的一部分。通过构建组合量子门可以大大简化量子电路的构建难度，也可以提高代码的复用性。

``` python
from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, H, CX

# 构建组合量子门
cg = CompositeGate()
CX | cg([0, 1])
CX | cg([1, 2])

# 构建 5 比特GHZ态量子电路
circuit = Circuit(5)
H | circuit(0)
cg | circuit([0, 1, 2])
cg | circuit([2, 3, 4])
```

### 动态量子电路

QuICT 构建了 Trigger 类，来支持量子电路的动态构建。与普通量子门不同的是， Trigger 类支持根据当前状态下的量子比特测量结果，运行不同的量子电路。

``` python
from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, H, CX
from QuICT.core.operator import Trigger

cir = Circuit(3)
H | cir(0)
CX | cir([0, 1])

# 构建组合量子门，测试用示例
cgate0 = CompositeGate()
CX | cgate0([0, 1])
cgate1 = CompositeGate()
CX | cgate1([1, 0])

# 构建 Trigger
state_mapping = {
    0: cgate0,      # 当量子比特测量值为0时，执行的电路
    1: cgate1       # 当量子比特测量值为1时，执行的电路
}

trigger = Trigger(
    targets=1,      # 量子比特数量
    state_gate_mapping=state_mapping    # 量子比特测量结果所对应的量子电路或者量子门
)

trigger | cir(1)
```

``` python

# 目标比特测量值为0的模拟结果
[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]

# 目标比特测量值为1的模拟结果
[0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
```


## 量子电路的矩阵表示

对于每一个量子电路，都可以由一个 $2^n \times 2^n$ 的酉矩阵所表示。

``` python
circuit.matrix()
```

``` python
[[0.70710678+0.j 0.        +0.j 0.        +0.j ... 0.        +0.j
  0.        +0.j 0.        +0.j]
 [0.        +0.j 0.70710678+0.j 0.        +0.j ... 0.        +0.j
  0.        +0.j 0.        +0.j]
 [0.        +0.j 0.        +0.j 0.        +0.j ... 0.        +0.j
  0.        +0.j 0.        +0.j]
 ...
 [0.        +0.j 0.        +0.j 0.        +0.j ... 0.        +0.j
  0.        +0.j 0.        +0.j]
 [0.        +0.j 0.70710678+0.j 0.        +0.j ... 0.        +0.j
  0.        +0.j 0.        +0.j]
 [0.70710678+0.j 0.        +0.j 0.        +0.j ... 0.        +0.j
  0.        +0.j 0.        +0.j]]
```

## 量子电路可视化

QuICT 支持量子电路可视化，可以通过 `Circuit.draw()` 实现。

``` python
circuit.draw()
```

<figure markdown>
![circuit_demo](../assets/images/tutorials/circuit_demo.jpg){:width="500px"}
</figure>

## 量子电路与 OPENQASM

QuICT 支持与 OPENQASM 之间的相互转换。

!!! note
    目前暂不支持 Unitary 门的 OPENQASM 转换。

``` python
from QuICT.tools.interface import OPENQASMInterface

# 将 QASM 文件转换为 Circuit
qasm_circuit = OPENQASMInterface.load_file("/path/to/qasm/file").circuit

# 将 Circuit 转换为 QASM
qasm_circuit.qasm(output_file="/path/to/store/qasm/file")
```

## 量子电路模拟

QuICT 通过使用 Simulator 来进行量子电路模拟，目前实现了三种模拟器。模拟器支持量子电路和初始状态向量作为输入，输出经过量子电路之后的状态向量，通常为一个 $2^n$ 的复数向量，如果模拟器选择使用密度矩阵模拟器，输出则会是密度矩阵，为一个 $2^n \times 2^n$ 的复数矩阵。

``` python
from QuICT.simulation.state_vector import StateVectorSimulator

# 使用上方构建的 5 比特GHZ态量子电路
circuit = ...

# 使用状态向量模拟器来进行量子电路模拟
simulator = StateVectorSimulator()
state_vector = simulator.run(circuit)
```

``` python
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j
 0.        +0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

```python
counts = simulator.sample(shots=3000)   # 3000次重复模拟实验结果
```

``` python
[1484, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1516]
```