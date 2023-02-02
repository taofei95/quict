# 基于图学习及强化学习的量子电路映射算法

## 术语定义

| 名称     | 含义                                                   |
| :------- | :----------------------------------------------------- |
| 拓扑     | 实际的量子电路中允许的双比特门连接性关系               |
| 逻辑电路 | 待映射的量子电路，其双比特门连接性可能并不满足拓扑约束 |
| 物理电路 | 实际可执行的量子电路，其双比特门关系必须满足拓扑约束   |

## 算法描述

电路中的量子门作用在同一个比特上时，必须按照特定的先后顺序执行。我们可以用一个有向无环图来表示这种先后顺序。由于在映射问题中，我们着重考虑双比特门，因此只将双比特门的依赖关系构建出有向无环图。图中的每个节点代表电路中的一个双比特门，图中的有向边 $u \rightarrow v$ 表示双比特门 $u$ 必须先于双比特门 $v$ 被执行。每个双比特门的标号，在电路中的位置等信息，被结合在一起处理为某种标记，与该双比特门对应的节点相关连。在这样的表示方法下，量子电路映射的过程可以概括如下：

1. 初始化物理电路为空，逻辑电路为待映射的电路，构造逻辑电路的有向无环图表示
2. 如果此时逻辑电路中存在部分无需任何操作就可以直接放入物理电路，那么直接将这部分双比特门转移至物理电路
3. 在有向无环图表示中删去上一步中删除的双比特门的节点（如果存在），更新节点标号
4. 此时如果逻辑电路已经被删空，则算法结束
5. 将有向无环图表示送入图神经网络，得到图的表示向量
6. 将图表示送入 DQN，它根据拓扑约束给出一个合法的物理电路中的双比特门位置
7. 在物理电路中选择该位置插入 SWAP 门
8. 因为 SWAP 的插入，有向无环图的标号需要更新
9. 跳转至 2 步骤，重复执行

### AI 推理

由于 DQN 需要基于拓扑执行决策，每个拓扑对应的网络略有不同，因此针对不同拓扑执行推理时，需要加载不同的模型文件。我们对几种常见拓扑提供了预训练的模型，可以直接用于映射算法。在映射算法初始化时，传入拓扑描述，以及模型路径文件夹（默认为 `./model`），算法会加载路径文件夹下与拓扑同名的模型描述文件。此时直接调用 `execute` 方法即可执行映射算法。

```python
from os import path as osp

from QuICT.core import Circuit, Layout
from QuICT.core.gate import *

from QuICT.qcda.mapping.ai.rl_mapping import RlMapping


def test_main():
    layout_path = osp.join(osp.dirname(__file__), "../layout")
    layout_path = osp.join(layout_path, "grid_3x3.json")
    layout = Layout.load_file(layout_path)
    # It will load model with the same name from model path.
    # If there's no model, you can try train first.
    mapper = RlMapping(layout=layout)
    circ = Circuit(9)
    circ.random_append(random_params=True)
    circ.draw(filename="before_mapping")
    mapped_circ = mapper.execute(circ)
    mapped_circ.draw(filename="mapped_circ")


if __name__ == "__main__":
    test_main()
```

### AI 训练

执行 `./QuICT/qcda/mapping/ai/train/train_rl.py` 脚本即可。`Trainer` 类需要一个 `TrainConfig` 实例进行实例化。只需向 `TrainConfig` 写入拓扑结构等关键信息，`Trainer` 就能自动开启 DQN 的训练过程并记录下最优模型。
