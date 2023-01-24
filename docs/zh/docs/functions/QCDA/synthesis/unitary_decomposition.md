# 酉矩阵分解

酉矩阵分解 (Unitary Decomposition) 算法针对给定的酉矩阵返回对应的量子电路。应用这一方法原则上可以实现对量子态的任意可容许操作，但其电路规模和深度均较高，请斟酌使用。

## 使用例

此部分具体原理请参见对应的原论文。

Shende, V.V., Bullock, S.S., & Markov, I.L. (2006). Synthesis of quantum-logic circuits. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 25, 1000-1010.

Drury, B., & Love, P.J. (2008). Constructive quantum Shannon decomposition from Cartan involutions. Journal of Physics A: Mathematical and Theoretical, 41, 395305.

``` python
from QuICT.qcda.synthesis import UnitaryDecomposition

UD = UnitaryDecomposition()
gates, _ = UD.execute(mat)
```

所得`gates`即对应于酉矩阵`mat`的量子电路，这里以`CompositeGate`的形式返回。
