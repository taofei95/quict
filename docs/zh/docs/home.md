---
hide:
  - navigation
---


<figure markdown>
![LOGO](/assets/images/home/quact.png)
</figure>

# **QuICT** 平台简介

**QuICT** (*Quantum Computer of Institute of Computing Technology*) 是一个开源量子计算操作平台。目前 **QuICT** 已包含6种常见指令集以及20余种量子门，可进行动态量子电路构建。已实现3种不同类型的量子电路模拟器，并且都有对CPU/GPU的支持和加速，通过密度矩阵模拟器可进行含噪声量子电路仿真模拟。设计并实现了 **QCDA** (*Quantum Circuit Design Automation*) 量子电路辅助设计模块，包括量子初态制备、指令集转换、酉矩阵合成、量子电路优化和映射等功能。算法方面，实现了*shor*、*grover*、*qae*等常见量子算法，可进行因数分解、SAT问题求解等；与此同时，也在 **QuICT_ML** 库里实现了QML领域内的 *QAOA* 和 *QNN* 算法，支持图求解最大割问题和MNIST手写数字图片二分类。 **QuICT**同时也构建了量子算法电路库和针对量子计算机的性能基准测试，通过设计不同的量子电路赛道，来实现针对量子机性能的全方位基准测试。

<figure markdown>
![QuICT](/assets/images/home/quict.png){:width="500px"}
</figure>

## 主要模块

- QuICT.algorithm: 包含多种常见量子算法，例如 shor, grover, cnf, vqa等。
- QuICT.core: 包含构建电路所需的组件，量子比特，量子门，量子电路等。
- QuICT.qcda: 量子电路合成、优化和映射
- QuICT.simulation: 量子电路模拟器，支持 Unitary、StateVector、DensityMatrix。
- QuICT.tools: 辅助开发模块，包括画图、OPENQASM转换、量子电路库以及Benchmark等。

## 更多功能

- QuICT_ML: 包含多种机器学习相关的量子算法库，比如 QAOA、VQE、以及基于强化学习的量子电路映射算法。
- QuICT_SIM: 量子电路模拟库，包含一个基于CPU的更高效快速的状态向量模拟器，以及多节点全振幅模拟器（暂未开源）。

## 相关链接

- QuICT 代码库：<https://gitee.com/quictucas/quict>
- QuICT_ML 代码库： <https://edu.gitee.com/quictucas/repos/quictucas/quict-ml/sources>
- QuICT_SIM 代码库： <https://edu.gitee.com/quictucas/repos/quictucas/quict-sim/sources>
- 文档：<http://10.25.0.56:8800/>
- Pypi：<https://pypi.org/project/quict/>
- UI (测试中)：<http://49.235.108.172:8080/>