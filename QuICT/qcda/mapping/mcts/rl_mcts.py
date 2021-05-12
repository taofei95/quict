from ..mcts.mcts import *
from ..mcts_cpp.mcts_wrapper import RLMCTSTreeWrapper


class RLMCTS(MCTS):
    def __init__(self, play_times: int = 1, gamma: float = 0.7, Gsim: int = 30, num_of_swap_gates: int = 10,
                 num_of_process: int = 2, bp_mode: int = 0, extended: bool = False,
                 Nsim: int = 500, selection_times: int = 40, c=20, graph_name: str = None, virtual_loss: float = 10,
                 is_generate_data: bool = False, with_predictor: bool = False,
                 threshold_size: int = 150, experience_pool: ExperiencePool = None, model_file_path: str = None,
                 device: torch.device = torch.device("cuda"), device_id: int = 0, **params):
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
        self._is_generate_data = is_generate_data
        self._coupling_graph = get_coupling_graph(graph_name=graph_name)
        self._experience_pool = experience_pool
        if device == torch.device("cpu"):
            assert (device_id < 3 and device_id >= 0)
            d = device_id + 1
        elif device == torch.device("cuda"):
            d = 0
        self._mcts_wrapper = RLMCTSTreeWrapper(is_generate_data=is_generate_data, threshold_size=threshold_size,
                                               gamma=self._gamma, c=self._c,
                                               virtual_loss=virtual_loss,
                                               bp_mode=bp_mode, num_of_process=num_of_process,
                                               size_of_subcircuits=self._Gsim, num_of_iterations=self._selection_times,
                                               num_of_swap_gates=num_of_swap_gates, num_of_playout=self._Nsim,
                                               num_of_edges=self._coupling_graph.num_of_edge,
                                               coupling_graph=self._coupling_graph.adj_matrix,
                                               distance_matrix=self._coupling_graph.distance_matrix,
                                               edge_label=self._coupling_graph.label_matrix,
                                               feature_matrix=self._coupling_graph.node_feature,
                                               model_file_path=model_file_path.encode("utf-8"),
                                               device=d)

    def search(self, logical_circuit: Circuit, init_mapping: List[int], extended: bool = False):
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
        self._gate_index = []
        self._physical_circuit: List[BasicGate] = []
        qubit_mask = np.zeros(self._coupling_graph.size, dtype=np.int32) - 1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.

        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit

        front_layer_ = [self._circuit_dag.index[i] for i in self._circuit_dag.front_layer]
        qubit_mask_ = [self._circuit_dag.index[i] if i != -1 else -1 for i in qubit_mask]
        print(front_layer_)
        print(qubit_mask_)
        self._mcts_wrapper.load_data(num_of_logical_qubits=logical_circuit.circuit_width(),
                                     num_of_gates=self._circuit_dag.size, circuit=self._circuit_dag.node_qubits,
                                     dependency_graph=self._circuit_dag.compact_dag,
                                     qubit_mapping=init_mapping, qubit_mask=qubit_mask_, front_layer=front_layer_)

        self._cur_node = MCTSNode(circuit_dag=self._circuit_dag, coupling_graph=self._coupling_graph,
                                  front_layer=self._circuit_dag.front_layer, qubit_mask=qubit_mask.copy(),
                                  cur_mapping=init_mapping.copy())

        self._add_initial_single_qubit_gate(node=self._cur_node)
        self._add_executable_gates(node=self._cur_node)
        res = self._mcts_wrapper.search(extended)

        if self._is_generate_data:
            num, adj, qubits, padding_mask, swap_gate, value, action_prob = self._mcts_wrapper.get_data()
            self._experience_pool.extend(adj=adj, qubits=qubits, action_probability=action_prob,
                                         value=value, circuit_size=padding_mask, swap_label=swap_gate, num=num)

        return res
