# 算法基准测试

## Shor因数分解算法

用Shor因数分解算法的变体中的HRS_zip变体（电路宽度12~20）和BEA_zip变体（电路宽度13~21），输入5~9位数，获取order-finding部分的电路后执行因数分解的运行时间对比。

<figure markdown>

![algorithm benchmark](../assets/images/QuICTbenchmark/algorithm_benchmark/QuICT_shor_algorithm_test.png)

</figure>

## Grover搜索算法

构建5~8变量的MCToracle电路，运行Grover搜索算法得到搜索结果的运行时间对比。

<figure markdown>

![algorithm benchmark](../assets/images/QuICTbenchmark/algorithm_benchmark/QuICT_grover_algorithm_test.png)

</figure>

## 最大割算法

根据哈密顿量初始化QAOA量子神经网络（4层量子电路层数），并开始迭代训练，通过量子测量对最终获得的量子态进行多次采样，并统计出现的比特串的概率分布。记录训练过程中最大割算法运行的时间对比。

<figure markdown>

![algorithm benchmark](../assets/images/QuICTbenchmark/algorithm_benchmark/QuICT_maxcut_test.png)

</figure>

## 量子游走搜索算法

在3~8维度的超立方体上执行多节点的量子游走搜索算法(QWS)算法，即同时标记节点3、4、6、8，对于给定的黑盒硬币C，对未标记节点的对应状态应用硬币C0，对标记节点的对应状态应用硬币C1。根据抛硬币结果执行移动操作的运行的时间对比。

<figure markdown>

![algorithm benchmark](../assets/images/QuICTbenchmark/algorithm_benchmark/QuICT_quantum_walk_search_test.png)

</figure>
