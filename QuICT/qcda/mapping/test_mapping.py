from typing import List
from QuICT.core.circuit import * 
from QuICT.core.layout import *
from QuICT.qcda.mapping import *

#输入一个量子电路实例与物理拓扑图
logical_circuit: Circuit 
physical_device_layout: Layout
#初始量子比特映射为可选项,不给定的情况下由算法指定
init_mapping: List[int] 
#算法输出一个物理可执行的电路
physical_circuit = Mapping.run(circuit = logical_circuit,
                               lyout = physical_device_layout,
                               init_mapping = init_mapping)