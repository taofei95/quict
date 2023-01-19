# Grover搜索算法

Grover搜索算法是一种无结构量子搜索算法，对于单个目标的情况，其查询复杂度是$O(\sqrt{N})$，其中$N$是状态空间大小。

这里实现了教科书版本[<sup>[1]</sup>](#refer1)的Grover算子，此外还支持多解情况，支持bit-flip和phase-flip oracle。当解的数目所占比例较大（超过一半）时，Grover迭代次数为0，算法退化为均匀随机采样。

Grover搜索算法的实际运行时间取决于谕示（oracle）电路的复杂程度。对于20个变量，解数目比例为$2.2\times10^{-3}$的SAT问题，算法在单块GPU上可以在一小时内完成。

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
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
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
grover = Grover(simulator=ConstantStateVectorSimulator())
result = grover.run(n, k, oracle)
print(result)
```

更多的例子见`example/demo/tutorial_grover.ipynb`和`unit_test/algorithm/quantum_algorithm/grover_unit_test.py`。

## 参考文献

<div id="refer1"></div>

<font size=3>
[1] Nielsen, M. A., & Chuang, I. L. (2019). *Quantum computation and quantum information*. Cambridge Cambridge University Press.
</font>