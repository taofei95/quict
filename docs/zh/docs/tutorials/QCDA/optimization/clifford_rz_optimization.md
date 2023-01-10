# Clifford+Rz优化

Clifford+Rz优化 (Clifford+Rz Optimization) 是一种优化大规模Clifford+Rz电路的启发式算法 [<sup>[1]</sup>](#refer1)。它有两种优化level——light和heavy。Light level的时间复杂度为 $O(g^2)$，其中 $g$ 是电路中的门数量。Heavy level的时间复杂度为 $O(g^3)$，它可能可以消去比light level更多的CNOT门。

除了Clifford+Rz门集合，这个优化算法也支持CCX/CCZ门，它可以贪心地将CCX/CCZ门分解成Clifford+Rz电路，并优化电路规模。电路中不支持的门类型将不会被改动。

## 基本用法

`CliffordRzOptimization` 位于`QuICT.qcda.optimization.clifford_rz_optimization`，支持三个可选的初始化参数：

1.  `level`: 优化级别，`'light'` 或者 `'heavy'`， 默认为 `'light'`。
2.  `optimize_toffoli`: 是否优化CCX/CCZ门，默认为 `True`。如果为 `True`，优化器会将CCX/CCZ门分解成Clifford+Rz电路；如果为 `False`，优化器不会改变电路中的CCX/CCZ门。
3.  `verbose`: 是否输出优化的具体过程信息，默认为 `False`。

对于一个待优化的电路 `circ`，首先实例化一个优化器 `CRO = CliffordRzOptimization()`（根据需要传入参数），然后执行 `CRO.execute(circ)` 得到优化后的电路。

## 代码示例

生成一个随机电路，然后使用Clifford+Rz优化算法。

```python
from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.qcda.optimization import CliffordRzOptimization

typelist = [GateType.x, GateType.cx, GateType.h, GateType.s,
            GateType.t, GateType.sdg, GateType.tdg, GateType.rz]

if __name__ == '__main__':

    # generate a random 5-qubit circuit using gates in typelist
    circuit = Circuit(5)
    circuit.random_append(100, typelist=typelist)
    circuit.draw(filename='0.jpg')

    # instantiate a optimizer
    CRO = CliffordRzOptimization()

    # optimize the circuit
    circ_optim = CRO.execute(circuit)
    circ_optim.draw(filename='1.jpg')
```

随机电路：

![circuit before](../../../assets/images/tutorials/QCDA/optimization/cro_0.jpg)

优化后的电路：

![circuit after](../../../assets/images/tutorials/QCDA/optimization/cro_1.jpg)

---

## 参考文献

<div id="refer1"></div>
<font size=3>
[1] Nam, Y., Ross, N.J., Su, Y. et al. Automated optimization of large quantum circuits with continuous parameters. npj Quantum Inf 4, 23 (2018). [https://doi.org/10.1038/s41534-018-0072-4](https://doi.org/10.1038/s41534-018-0072-4)
</font>

---