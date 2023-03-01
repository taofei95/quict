# QCDA基准测试

## 映射基准测试

构建3×3网状结构拓扑、9量子比特、36~180量子门数的量子电路，执行SABRE、MCTS、RL三种映射方法后量子电路映射后量子电路中新增的交换门门数进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/mapping_benchmark/QuICT_mapping_test_grid.png)

</figure>

构建5比特T型结构拓扑、5量子比特、36~180量子门数的量子电路，执行SABRE、MCTS、RL三种映射方法后量子电路映射后量子电路中新增的交换门门数进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/mapping_benchmark/QuICT_mapping_test_T.png)

</figure>

## 优化基准测试

### Clifford+Rz电路优化

构建4~20量子比特、80~400量子门数、克里弗电路指令集构造的的量子电路，执行Clifford+Rz电路优化前后的门数、深度、双比特门门数进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/optimization_benchmark/QuICT%20Clifford_Rz_Optimization%20test.png)

</figure>

### 无辅助比特合成电路优化

构建4~20量子比特、80~400量子门数、单一可控非门构造的的量子电路，执行无辅助比特合成电路优化前后的门数、深度、双比特门门数进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/optimization_benchmark/QuICT%20cnot_without_ancilla%20test.png)

</figure>

### 交换优化

构建4~20量子比特、80~400量子门数、随机指令集构造的的量子电路，执行无辅助比特合成后电路优化前后的门数、深度、双比特门门数进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/optimization_benchmark/QuICT%20Commutative_Optimization%20test.png)

</figure>

### Clifford电路与Symbolic优化

构建4~20量子比特、80~400量子门数、随机指令集构造的的量子电路，执行Clifford电路与Symbolic优化前后的门数、深度、双比特门门数进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/optimization_benchmark/QuICT%20Symbolic_Clifford_Optimization%20test.png)

</figure>

### 模板匹配优化

构建4~20量子比特、80~400量子门数、克里弗电路指令集构造的的量子电路，执行电路模板匹配后的前后的门数、深度、双比特门门数进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/optimization_benchmark/QuICT%20Template_Optimization%20test.png)

</figure>

## 合成基准测试

### 门转换

构建5量子比特、50量子门数、随机指令集构造的的量子电路，选择六种指令集对每个电路执行门转换电路优化前后的门数、深度进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/synthesis_benchmark/QuICT_gate_transform_test.png)

</figure>

### 量子态制备

构建4~8量子比特的态向量构造的的量子电路，执行量子态制备算法后的原量子电路和返回的量子电路的门数、深度进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/synthesis_benchmark/QuICT_Quantum_state_preparation_test.png)

</figure>

### 酉矩阵分解

构建4~8量子比特的酉矩阵构造的的量子电路，执行酉矩阵分解后的原量子电路和返回的量子电路的门数、深度进行对比。

<figure markdown>

![qcda benchmark](../assets/images/QuICTbenchmark/qcda_benchmark/synthesis_benchmark/QuICT_unitary_decomposition_test.png)

</figure>