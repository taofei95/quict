#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 9:41 下午
# @Author  : Han Yu
# @File    : _circuit.py

from ._qubit import Qubit, Qureg
from QuICT.exception import TypeException, ConstException
import numpy as np
import random

circuit_id = 0

"""
电路类
"""
class Circuit(object):
    """
    类的属性
    """

    # 常量锁
    @property
    def const_lock(self) -> bool:
        return self.__const_lock

    @const_lock.setter
    def const_lock(self, const_lock):
        self.__const_lock = const_lock

    # 电路的id
    @property
    def id(self) -> int:
        return self.__id

    @id.setter
    def id(self, id):
        if not self.const_lock:
            self.__id = id
        else:
            raise ConstException(self)

    # 电路的name
    @property
    def name(self) -> int:
        return self.__name

    @name.setter
    def name(self, name):
        if not self.const_lock:
            self.__name = name
        else:
            raise ConstException(self)

    # 所占据的wire
    @property
    def qubits(self) -> Qureg:
        return self.__qubits

    @qubits.setter
    def qubits(self, qubits):
        if not self.const_lock:
            self.__qubits = qubits
        else:
            raise ConstException(self)

    # 电路门
    @property
    def gates(self) -> list:
        return self.__gates

    @gates.setter
    def gates(self, gates):
        if not self.const_lock:
            self.__gates = gates
        else:
            raise ConstException(self)

    # 电路拓扑
    @property
    def topology(self) -> list:
        return self.__topology

    @topology.setter
    def topology(self, topology):
        if not self.const_lock:
            self.__topology = topology
        else:
            raise ConstException(self)

    # 电路保真度
    @property
    def fidelity(self) -> float:
        return self.__fidelity

    @fidelity.setter
    def fidelity(self, fidelity):
        if fidelity is None:
            self.__fidelity = None
            return
        if not isinstance(fidelity, float) or fidelity < 0 or fidelity >= 1.0:
            raise Exception("保真度应为0到1之间的常数")
        if not self.const_lock:
            self.__adjust_fidelity = True
            self.__fidelity = fidelity
        else:
            raise ConstException(self)

    @staticmethod
    def getRandomList(l, n):
        _rand = [i for i in range(n)]
        for i in range(n - 1, 0, -1):
            do_get = random.randint(0, i)
            _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
        return _rand[:l]

    def random(self, rand_size = 10, typeList = None):
        """
        生成一个随机电路
        :param rand_size: 电路size
        :param typeList: 门类型（默认包含CX、ID、Rz、CY、CRz、CH）
        :return:
        """
        from ._gate import GateBuilder, GateType
        if typeList is None:
            typeList = [GateType.CX, GateType.ID, GateType.Rz, GateType.CY, GateType.CRz, GateType.CH]
        qubit = len(self.qubits)
        for _ in range(rand_size):
            rand_type = random.randrange(0, len(typeList))
            GateBuilder.setGateType(typeList[rand_type])

            targs = GateBuilder.getTargsNumber()
            cargs = GateBuilder.getCargsNumber()
            pargs = GateBuilder.getParamsNumber()

            tclist = self.getRandomList(targs + cargs, qubit)
            if targs != 0:
                GateBuilder.setTargs(tclist[:targs])
            if cargs != 0:
                GateBuilder.setCargs(tclist[targs:])
            if pargs != 0:
                params = []
                for _ in range(pargs):
                    params.append(random.uniform(0, 2 * np.pi))
                GateBuilder.setPargs(params)
            gate = GateBuilder.getGate()
            self.gates.append(gate)

    def index_for_qubit(self, qubit, ancilla = None) -> int:
        """
        :param qubit: 需要查询的qubit
        :param ancilla: 不计算的辅助位
        :return: 索引值
        :raise 传入的参数不是qubit，或者不在该tangle中
        """
        if not isinstance(qubit, Qubit):
            raise TypeException("Qubit", now=qubit)
        if ancilla is None:
            ancilla = []
        enterspace = 0
        for i in range(len(self.qubits)):
            if i in ancilla:
                enterspace += 1
            elif self.qubits[i].id == qubit.id:
                return i - enterspace
        raise Exception("传入的qubit不在该电路中或为辅助位")

    def matrix_product_to_circuit(self, gate) -> np.ndarray:
        """
        :return: 返回从矩阵直积到整个矩阵
        """

        q_len = len(self.qubits)
        n = 1 << len(self.qubits)

        new_values = np.zeros((n, n), dtype=np.complex)
        targs = gate.targs
        cargs = gate.cargs
        if not isinstance(targs, list):
            targs = [targs]
        if not isinstance(cargs, list):
            cargs = [cargs]
        targs = np.append(np.array(cargs, dtype=int).ravel(), np.array(targs, dtype=int).ravel())
        targs = targs.tolist()
        # targs.reverse()
        xor = (1 << q_len) - 1
        if not isinstance(targs, list):
            raise Exception("未知错误")
        matrix = gate.compute_matrix().reshape(1 << len(targs), 1 << len(targs))
        datas = np.zeros(n, dtype=int)
        # print(targs)
        for i in range(n):
            nowi = 0
            for kk in range(len(targs)):
                k = q_len - 1 - targs[kk]
                if (1 << k) & i != 0:
                    nowi += (1 << (len(targs) - 1 - kk))
            datas[i] = nowi
        # print(datas)
        for i in targs:
            xor = xor ^ (1 << (q_len - 1 - i))
        for i in range(n):
            nowi = datas[i]
            for j in range(n):
                nowj = datas[j]
                if (i & xor) != (j & xor):
                    continue
                new_values[i][j] = matrix[nowi][nowj]
        return new_values

    def circuit_length(self):
        """
        :return: 电路比特数
        """
        return len(self.qubits)

    def circuit_size(self):
        """
        :return: 电路门数
        """
        return len(self.gates)

    def circuit_count_2qubit(self):
        """
        :return: 双比特门个数
        """
        count = 0
        for gate in self.gates:
            if gate.controls + gate.targets == 2:
                count += 1
        return count

    def circuit_count_1qubit(self):
        """
        :return: 单比特门个数
        """
        count = 0
        for gate in self.gates:
            if gate.controls + gate.targets == 1:
                count += 1
        return count

    def circuit_count_gateType(self, gateType):
        """
        :param gateType: 要统计的门类型
        :return: 该类型门的总个数
        """
        count = 0
        for gate in self.gates:
            if gate.type == gateType:
                count += 1
        return count

    def circuit_depth(self, gateTypes = None):
        """
        :param gateTypes:  list, 表示统计进深度的门类型
        :return: 电路深度
        """
        layers = []
        for gate in self.gates:
            if gateTypes is None or gate.type in gateTypes:
                now = set(gate.cargs) | set(gate.targs)
                for i in range(len(layers) - 1, -2, -1):
                    if i == -1 or len(now & layers[i]) > 0:
                        if i + 1 == len(layers):
                            layers.append(set())
                        layers[i + 1] |= now
        return len(layers)

    def __init__(self, wires):
        """
        :param wires: 电路位数
        """
        self.const_lock = False
        global circuit_id
        self.id = circuit_id
        self.name = "电路" + str(self.id)
        circuit_id = circuit_id + 1
        self.__idmap = {}
        self.qubits = Qureg()
        for idx in range(wires):
            qubit = Qubit(self)
            self.qubits.append(qubit)
            self.__idmap[qubit.id] = idx
        self.gates = []
        self.__queue_gates = []
        self.topology = []
        self.fidelity = None
        self.__adjust_fidelity = False

    def __call__(self, other: object) -> Qureg:
        """
        :param other:
        1) int
        2) list
        3) tuple
        :return: Qureg
        :exception: 传入类型错误或索引范围错误
        """

        # 处理int
        if isinstance(other, int):
            if other < 0 or other >= len(self.qubits):
                raise IndexLimitException(len(self.qubits), other)
            return Qureg(self.qubits[other])

        # 处理tuple
        if isinstance(other, tuple):
            other = list(other)
        # 处理list
        if isinstance(other, list):
            if len(other) != len(set(other)):
                raise IndexDuplicateException(other)
            qureg = Qureg()
            for element in other:
                if not isinstance(element, int):
                    raise TypeException("int", element)
                if element < 0 or element >= len(self.qubits):
                    raise IndexLimitException(len(self.qubits), element)
                qureg.append(self.qubits[element])
            return qureg

        raise TypeException("int或list或tuple", other)

    def set_flush_gates(self, gates):
        self.gates = gates.copy()
        self.__queue_gates = gates.copy()

    def __add_qubit_gate__(self, gate, qubit):
        """
        :param gate:    添加的gate实例
        :param qubit:   作用的qubit
        :return:
        """
        self.gates.append(gate)
        self.__queue_gates.append(gate)
        gate.targs = [self.__idmap[qubit.id]]

    def __add_qureg_gate__(self, gate, qureg):
        """
        :param gate:    添加的gate实例
        :param qureg:   作用的qureg
        :return:
        """
        self.gates.append(gate)
        self.__queue_gates.append(gate)
        gate.cargs = [self.__idmap[qureg[idx].id] for idx in range(gate.controls)]
        gate.targs = [self.__idmap[qureg[idx].id] for idx in range(gate.controls, gate.controls + gate.targets)]

    def __inner_add_topology__(self, topology):
        """
        添加拓扑逻辑
        :param topology:
            1) tuple<qureg/qubit/int>
        :raise TypeException
        """
        if not isinstance(topology, tuple):
            raise TypeException("tuple<qureg(len = 1)/qubit/int>或list<tuple<qureg(len = 1)/qubit/int>>", topology)
        if len(topology) != 2:
            raise Exception("输出的tuple应该有2")
        item1 = topology[0]
        item2 = topology[1]
        if isinstance(item1, Qureg):
            if len(item1) != 1:
                item1 = item1[0]
        if isinstance(item1, Qubit):
            item1 = self.__idmap[item1]
        if not isinstance(item1, int):
            raise TypeException("tuple<qureg(len = 1)/qubit/int>或list<tuple<qureg(len = 1)/qubit/int>>", topology)
        if isinstance(item2, Qureg):
            if len(item2) != 1:
                item2 = item2[0]
        if isinstance(item2, Qubit):
            item2 = self.__idmap[item2]
        if not isinstance(item2, int):
            raise TypeException("tuple<qureg(len = 1)/qubit/int>或list<tuple<qureg(len = 1)/qubit/int>>", topology)
        self.topology.append((item1, item2))

    def add_topology(self, topology):
        """
        添加拓扑逻辑
        :param topology:
            1) tuple<qureg(len = 1)/qubit/int>
            2) list<tuple<qureg(len = 1)/qubit/int>>
        :raise TypeException
        """
        if isinstance(topology, list):
            for item in topology:
                self.__inner_add_topology__(item)
        self.__inner_add_topology__(topology)

    def add_topology_complete(self, qureg : Qureg):
        """
        添加完全图拓扑逻辑
        :param qureg: 在它们之间构建完全拓扑图
        """
        ids = []
        for qubit in qureg:
            ids.append(self.__idmap[qubit.id])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self.__inner_add_topology__((i, j))

    def clean_qubits_and_gates(self):
        for qubit in self.qubits:
            qubit.tangle_clear()
        self.__queue_gates = []
        self.gates = []

    def reset_initial_values(self):
        for qubit in self.qubits:
            qubit.tangle_clear()
        self.qubits.force_assign_random()
        self.__queue_gates = []
        self.gates = []

    def reset_initial_zeros(self):
        for qubit in self.qubits:
            qubit.tangle_clear()
        self.qubits.force_assign_zeros()
        self.__queue_gates = []
        self.gates = []

    def reset_all(self):
        for qubit in self.qubits:
            qubit.tangle_clear()
        self.__queue_gates = []
        self.gates = []

    def print_infomation(self):
        print("-------------------")
        print("比特位数:{}".format(self.circuit_length()))
        for gate in self.gates:
            gate.print_info()
        print("-------------------")

    def force_copy(self, other, force_copy = None):
        for qubit in self.qubits:
            qubit.tangle_clear()
        self.__queue_gates = []
        self.gates = []
        if force_copy is None:
            force_copy = [i for i in range(len(self.qubits))]
        self.qubits.force_copy(other.qubits[0].tangle, force_copy)

    def complete_flush(self):
        self.flush()
        self.gates = []

    def flush(self):
        """
        作用所有未处理的门
        :raise 给出了无法处理的门
        """
        for gate in self.__queue_gates:
            if gate.is_single():
                tangle = self.qubits[gate.targ].tangle
                if self.__adjust_fidelity:
                    tangle.deal_single_gate(gate, True, self.fidelity)
                else:
                    tangle.deal_single_gate(gate)
            elif gate.is_control_single():
                tangle0 = self.qubits[gate.carg].tangle
                tangle1 = self.qubits[gate.targ].tangle
                tangle0.merge(tangle1)
                tangle0.deal_control_single_gate(gate)
            elif gate.is_ccx():
                tangle0 = self.qubits[gate.cargs[0]].tangle
                tangle1 = self.qubits[gate.cargs[1]].tangle
                tangle2 = self.qubits[gate.targ].tangle
                tangle0.merge(tangle1)
                tangle0.merge(tangle2)
                tangle0.deal_ccx_gate(gate)
            elif gate.is_swap():
                tangle0 = self.qubits[gate.targs[0]].tangle
                tangle1 = self.qubits[gate.targs[1]].tangle
                tangle0.merge(tangle1)
                tangle0.deal_swap_gate(gate)
            elif gate.is_measure():
                tangle0 = self.qubits[gate.targ].tangle
                tangle0.deal_measure_gate(gate)
            elif gate.is_custom():
                targs = gate.targs
                if not isinstance(targs, list):
                    tangle = self.qubits[targs].tangle
                else:
                    tangle = self.qubits[targs[0]].tangle
                for i in range(1, gate.targets):
                    new_tangle = self.qubits[targs[i]].tangle
                    tangle.merge(new_tangle)
                tangle.deal_custom_gate(gate)
            elif gate.is_perm():
                targs = gate.targs
                if not isinstance(targs, list):
                    tangle = self.qubits[targs].tangle
                else:
                    tangle = self.qubits[targs[0]].tangle
                for i in range(1, gate.targets):
                    new_tangle = self.qubits[gate.targs[i]].tangle
                    tangle.merge(new_tangle)
                tangle.deal_perm_gate(gate)
            elif gate.is_shorInit():
                targs = gate.targs
                if not isinstance(targs, list):
                    tangle = self.qubits[targs].tangle
                else:
                    tangle = self.qubits[targs[0]].tangle
                for i in range(1, gate.targets):
                    new_tangle = self.qubits[gate.targs[i]].tangle
                    tangle.merge(new_tangle)
                tangle.deal_shorInitial_gate(gate)
            elif gate.is_controlMulPer():
                targs = gate.targs
                if not isinstance(targs, list):
                    tangle = self.qubits[targs].tangle
                else:
                    tangle = self.qubits[targs[0]].tangle
                for i in range(1, gate.targets):
                    new_tangle = self.qubits[gate.targs[i]].tangle
                    tangle.merge(new_tangle)
                new_tangle = self.qubits[gate.cargs[0]].tangle
                tangle.merge(new_tangle)
                tangle.deal_controlMulPerm_gate(gate)
            elif gate.is_reset():
                tangle0 = self.qubits[gate.targ].tangle
                tangle0.deal_reset_gate(gate)
            elif gate.is_barrier():
                pass
            else:
                raise Exception("给出了无法处理的门")
        self.__queue_gates = []

    def qasm(self):
        """
        输出电路对应的qasm文本，如果不合法，则输出"error"
        :return: 一个qasm文本，或者"error"
        """
        string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        cbits = 0
        for gate in self.gates:
            if gate.is_measure():
                cbits += 1
        string += "qreg q[{}];\n".format(self.circuit_length())
        if cbits != 0:
            string += "creg c[{}];\n".format(cbits)
        cbits = 0
        for gate in self.gates:
            if gate.is_measure():
                string += "measure q[{}] -> c[{}];\n".format(gate.targ, cbits)
                cbits += 1
            else:
                qasm = gate.qasm()
                if qasm == "error":
                    print("这个电路不能转化为合法的QASM")
                    gate.print_info()
                    return "error"
                string += gate.qasm()
        return string

    def draw_photo(self, filename = None, show_depth = True):
        """
        :param filename:    输出的文件名(不包含后缀)，默认为电路名称
        :param show_depth:  在图形化界面中展示同层信息
        :return:
        """
        from QuICT.drawer import PhotoDrawerModel
        if filename is None:
            filename = str(self.id) + '.jpg'
        elif '.' not in filename:
            filename += '.jpg'
        PhotoDrawer = PhotoDrawerModel()
        PhotoDrawer.run(self, filename, show_depth)

    def partial_prop(self, other):
        self.flush()
        if not isinstance(other, list):
            tangle = self.qubits[other].tangle
        else:
            tangle = self.qubits[other[0]].tangle
        for i in range(1, len(other)):
            new_tangle = self.qubits[other[i]].tangle
            tangle.merge(new_tangle)
        ids = []
        for index in other:
            ids.append(self.qubits[index].id)
        return tangle.partial_prop(ids)

    def __del__(self):
        for qubit in self.qubits:
            qubit.tangle_clear()
        self.gates = None
        self.__queue_gates = None
        self.qubits = None
        self.topology = None
        print("del circuit")

"""
索引范围错误
"""
class IndexLimitException(Exception):
    def __init__(self, wire, try_index):
        """
        :param wire: 索引位数
        :param try_index: 尝试使用的索引
        """
        Exception.__init__(self, "索引范围为0～{},但尝试索引了{}".format(wire, try_index))

"""
索引重复错误
"""
class IndexDuplicateException(Exception):
    def __init__(self, other):
        """
        :param other: 错误索引list或者tuple
        """
        Exception.__init__(self, "索引出现重复:{}".format(other))
