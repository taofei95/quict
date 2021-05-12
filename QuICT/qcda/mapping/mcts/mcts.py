from queue import deque

from QuICT.qcda.mapping.mcts_cpp import MCTSTreeWrapper
from QuICT.qcda.mapping.utility import *


class MCTS(object):
    @classmethod
    def _get_physical_gate(cls, gate: BasicGate, cur_mapping: List[int]) -> BasicGate:
        """
        Get the  physical gate of the given logical gate   
        """
        cur_gate = gate.copy()
        if cur_gate.is_single():
            target = cur_mapping[gate.targ]
            cur_gate.targs = int(target)
        elif cur_gate.is_control_single():
            control = cur_mapping[gate.carg]
            target = cur_mapping[gate.targ]
            cur_gate.cargs = int(control)
            cur_gate.targs = int(target)
        elif cur_gate.type() == GATE_ID['Swap']:
            target_0 = cur_mapping[gate.targs[0]]
            target_1 = cur_mapping[gate.targs[1]]
            cur_gate.targs = [int(target_0), int(target_1)]
        else:
            raise Exception("the gate type is not valid ")
        return cur_gate

    def __init__(self, play_times: int = 1, gamma: float = 0.8, Gsim: int = 30, num_of_swap_gates: int = 15,
                 num_of_process: int = 4, bp_mode: int = 0,
                 Nsim: int = 2, selection_times: int = 5000, c=20, graph_name: str = None,
                 coupling_graph: CouplingGraph = None, virtual_loss: float = 0, is_generate_data: bool = False,
                 threshold_size: int = 150, experience_pool: ExperiencePool = None, extended: bool = True,
                 with_predictor: bool = False, info: int = 0,
                 method: int = 0, major: int = 4, **params):
        """
        Params:
            paly_times: The repeated times of the whole search procedure for the circuit.
            gamma: The parameter measures the trade-off between the short-term reward and the long-term value.
            c: The parameter measures the trade-off between the exploitation and exploration in the upper confidence bound.
            Gsim: Size of the sub circuit that would be searched by the random simulation method.
            Nsim: The repeated times of the random simulation method.
            selection_times: The time of expansion and back propagation in the monte carlo tree search
        """
        self._play_times = play_times
        self._selection_times = selection_times
        self._call_back_threshold = 10
        self._gamma = gamma
        self._Gsim = Gsim
        self._Nsim = Nsim
        self._c = c
        self._major = major
        self._is_generate_data = is_generate_data
        self._num_of_swap_gate = num_of_swap_gates
        self._num_of_process = num_of_process
        self._virtual_loss = virtual_loss
        self._method = method
        self._extended = extended
        self._with_predictor = with_predictor
        self._bp_mode = bp_mode
        if graph_name is None:
            self._coupling_graph = coupling_graph
        else:
            self._coupling_graph = get_coupling_graph(graph_name=graph_name)

        self._experience_pool = experience_pool
        self._mcts_wrapper = MCTSTreeWrapper(major=major, method=method, info=info, extended=extended,
                                             with_predictor=with_predictor, is_generate_data=is_generate_data,
                                             threshold_size=threshold_size, gamma=self._gamma, c=self._c,
                                             virtual_loss=virtual_loss,
                                             bp_mode=bp_mode, num_of_process=num_of_process,
                                             size_of_subcircuits=self._Gsim, num_of_iterations=self._selection_times,
                                             num_of_swap_gates=num_of_swap_gates, num_of_playout=self._Nsim,
                                             num_of_edges=self._coupling_graph.num_of_edge,
                                             coupling_graph=self._coupling_graph.adj_matrix,
                                             distance_matrix=self._coupling_graph.distance_matrix,
                                             edge_label=self._coupling_graph.label_matrix,
                                             feature_matrix=self._coupling_graph.node_feature)

    @property
    def physical_circuit(self):
        """
        The physical circuit of the current root node
        """
        return self._physical_circuit

    @property
    def cur_node(self):
        """
        The root node of the monte carlo tree
        """
        return self._cur_node

    def search(self, logical_circuit: Circuit, init_mapping: List[int]):
        """
        The main process of the qubit mapping algorithm based on the monte carlo tree search. 
        Params:
            logical_circuit: The logical circuit to be transformed into the circuit compliant with the physical device, i.e., 
                            each gate of the transformed circuit are adjacent on the device.
            init_mapping: The initial mapping of the logical quibts to physical qubits.
            coupling_graph: The list of the edges of the physical device's graph.
        """

        self._num_of_executable_gate = 0
        self._logical_circuit_dag = DAG(circuit=logical_circuit, mode=Mode.WHOLE_CIRCUIT)
        self._circuit_dag = DAG(circuit=logical_circuit, mode=Mode.TWO_QUBIT_CIRCUIT)
        # print(self._circuit_dag.size)
        # print(self._logical_circuit_dag.size)
        self._gate_index = []
        self._physical_circuit: List[BasicGate] = []
        qubit_mask = np.zeros(self._coupling_graph.size, dtype=np.int32) - 1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.

        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit

        front_layer_ = [self._circuit_dag.index[i] for i in self._circuit_dag.front_layer]
        qubit_mask_ = [self._circuit_dag.index[i] if i != -1 else -1 for i in qubit_mask]
        # print(front_layer_)
        # print(qubit_mask_)
        self._mcts_wrapper.load_data(num_of_logical_qubits=logical_circuit.circuit_width(),
                                     num_of_gates=self._circuit_dag.size,
                                     circuit=self._circuit_dag.node_qubits,
                                     dependency_graph=self._circuit_dag.compact_dag,
                                     qubit_mapping=init_mapping, qubit_mask=qubit_mask_, front_layer=front_layer_)

        self._cur_node = MCTSNode(circuit_dag=self._circuit_dag, coupling_graph=self._coupling_graph,
                                  front_layer=self._circuit_dag.front_layer, qubit_mask=qubit_mask.copy(),
                                  cur_mapping=init_mapping.copy())
        res = self._mcts_wrapper.search()
        swap_label_list = self._mcts_wrapper.get_swap_gate_list()

        self._add_initial_single_qubit_gate(node=self._cur_node)
        self._add_executable_gates(node=self._cur_node)
        for swap_label in swap_label_list:
            self._cur_node.update_by_swap_gate(swap_label)
            self._physical_circuit.append(self._coupling_graph.get_swap_gate(swap_label))
            self._add_executable_gates(node=self._cur_node)

        assert (self._cur_node.is_terminal_node())

        if self._is_generate_data:
            num, adj, qubits, padding_mask, swap_gate, value, action_prob = self._mcts_wrapper.get_data()
            self._experience_pool.extend(adj=adj, qubits=qubits, action_probability=action_prob,
                                         value=value, circuit_size=padding_mask, swap_label=swap_gate, num=num)

        return res

    def _add_initial_single_qubit_gate(self, node: MCTSNode):
        """
        Add the single qubit gate in the initial front layer to the physical circuit
        """
        cur_mapping = node.cur_mapping
        sqg_stack = deque(self._logical_circuit_dag.front_layer)
        while len(sqg_stack) > 0:
            gate = sqg_stack.pop()
            if self._logical_circuit_dag[gate]['gate'].is_single():
                self._physical_circuit.append(self._get_physical_gate(gate=self._logical_circuit_dag[gate]['gate'],
                                                                      cur_mapping=cur_mapping))
                for suc in self._logical_circuit_dag.get_successor_nodes(vertex=gate):
                    if self._logical_circuit_dag[suc]['gate'].is_single():
                        sqg_stack.append(suc)

    def _add_executable_gates(self, node: MCTSNode):
        """
        Add  executable gates in the node to the physical circuit
        """
        execution_list = node.execution_list
        self._num_of_executable_gate = self._num_of_executable_gate + len(execution_list)

        for gate in execution_list:
            if self._logical_circuit_dag[gate]['gate'].is_single():
                raise Exception("There shouldn't exist single qubit gate in two-qubit gate circuit")
            else:
                self._add_gate_and_successors(gate)

    def _add_gate_and_successors(self, gate: int):
        """
        Add the current gate as well as  the gates succeed it to the physical circuit iff the succeeding gate 
        satisfies one of the following conditions: 
            1) is a single qubit gates 
            2) or is a two-qubit gate shares the same qubits as the gate 

        """
        cur_mapping = self._cur_node.cur_mapping
        sqg_stack = deque([gate])

        mark = set()
        while len(sqg_stack) > 0:
            gate_t = sqg_stack.pop()
            self._physical_circuit.append(self._get_physical_gate(gate=self._logical_circuit_dag[gate_t]['gate'],
                                                                  cur_mapping=cur_mapping))
            for suc in self._logical_circuit_dag.get_successor_nodes(vertex=gate_t):
                if self._is_valid_successor(suc, gate, mark):
                    sqg_stack.append(suc)

    def _is_valid_successor(self, gate_index: int, pre_gate_index: int, mark: Set[int]) -> bool:
        """
        Indicate whether the succeeding node can be excuted as soon as the  
        """
        gate = self._logical_circuit_dag[gate_index]['gate']
        pre_gate = self._logical_circuit_dag[pre_gate_index]['gate']
        if gate.is_single():
            return True
        elif gate.controls + gate.targets == 2:
            qubits = self._get_gate_qubits(gate)
            pre_qubits = self._get_gate_qubits(pre_gate)
            if (qubits[0] == pre_qubits[0] and qubits[1] == pre_qubits[1]) or (
                    qubits[0] == pre_qubits[1] and qubits[1] == pre_qubits[0]):
                if gate_index not in mark:
                    mark.add(gate_index)
                    return False
                else:
                    return True
            else:
                False
        else:
            raise Exception("The gate type is not supported")

    def _get_gate_qubits(self, gate: BasicGate) -> List[int]:
        """
        Get the qubits that the gate acts on 
        """
        if gate.is_control_single():
            return [gate.targ, gate.carg]
        elif gate.type() == GATE_ID['Swap']:
            return gate.targs
        else:
            raise Exception("The gate type %d is not supported" % (gate.type()))


def __str__(self):
    return str(self.__dict__)
