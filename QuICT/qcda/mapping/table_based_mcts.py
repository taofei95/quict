
from os import chmod
import time
import logging
import logging.config
from queue import Queue
from collections import deque
from enum import Enum

from networkx.generators.degree_seq import expected_degree_graph

from RL.experience_pool_v4 import ExperiencePool
from mcts_node import  *
from random_simulator.random_simulator import RandomSimulator


class TableBasedMCTS(MCTSBase):
    @classmethod
    def _get_physical_gate(cls, gate: BasicGate, cur_mapping: List[int])->BasicGate:
        """
        Get the  physical gate of the given logical gate   
        """
        if isinstance(gate, BasicGate):
            cur_gate = gate.copy()
            if cur_gate.is_single():
                target = cur_mapping[gate.targ]
                cur_gate.targs = target
            elif cur_gate.is_control_single():
                control = cur_mapping[gate.carg]
                target = cur_mapping[gate.targ]
                cur_gate.cargs = control
                cur_gate.targs = target
            elif cur_gate.type() == GATE_ID['Swap']:
                target_0 = cur_mapping[gate.targs[0]]
                target_1 = cur_mapping[gate.targs[1]]
                cur_gate.targs = [target_0, target_1]
            else:
                raise Exception("the gate type is not valid ")
        else:
            raise Exception("it is not the gate")
        return cur_gate
    
    @staticmethod
    def _fill_adj(array: np.ndarray):
        out = np.zeros(len(array)-1, dtype = array.dtype)
        for i in range(len(out)):
            if array[i+1] == -1:
                out[i] = array[0]
            else:
                out[i] = array[i+1]
        return out
   
    def __init__(self, play_times: int = 1, gamma: float = 0.7, Gsim: int = 20, size_threshold: int = 150, coupling_graph : str = None,
                 Nsim: int = 500, selection_times: int = 40, c = 20, mode: MCTSMode = MCTSMode.SEARCH, sim: SimMode = SimMode.MAX, rl: RLMode = RLMode.WARMUP, experience_pool: ExperiencePool = None, log_path: str = None, extended:bool = False ,**params):
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
        self._call_back_threshold = 15
        self._gamma = gamma
        self._Gsim = Gsim
        self._Nsim = Nsim
        self._c = c
        self._mode = mode
        self._sim = sim
        self._size_threshold = size_threshold
        self._experience_pool = experience_pool
        self._log_path =  log_path
        self._coupling_graph = get_coupling_graph(graph_name = coupling_graph)
        
        self._label_list = []
        self._num_list = []
        self._adj_list = []
        self._qubits_list = []
        self._value_list = []
        self._action_probability_list = []
        self._rs = None
        self._rl = rl
        self._extended = extended

        logging.config.dictConfig({"version":1,"disable_existing_loggers": False})
        # 第一步，创建一个logger
        self._logger = logging.getLogger("search")
        self._logger.setLevel(logging.ERROR)
        
        #第二步，创建一个handler，用于写入日志文件
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        if log_path is not None:
            fh = logging.FileHandler(self._log_path, mode='w')
            fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
            # 第三步，定义handler的输出格式
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR) 
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)
    
    @property
    def physical_circuit(self):
        """
        The physical circuit of the current root node
        """
        return self._physical_circuit

    @property
    def root_node(self):
        """
        The root node of the monte carlo tree
        """
        return self._root_node
    
    def _upper_confidence_bound(self, node: MCTSNode)->float:
        """
        The upper confidence bound of the node
        """
        # if node.visit_count == 0:
        #     return node.reward + node.value + self._c*100
        # else:
        #     return node.reward + node.value + self._c* np.sqrt(np.log(node.parent.visit_count) / node.visit_count)
        return node.reward + self._gamma * node.value + self._c* np.sqrt(np.log(node.parent.visit_count) / (node.visit_count+0.001))

    def _upper_confidence_bound_with_predictor(self, node: MCTSNode)->float:
        """
        The upper confidence bound with predictor of the node
        """
        return node.reward + self._gamma * node.value  + self._c* node.prob_of_edge *np.sqrt(node.parent.visit_count) / (node.visit_count+1) 

    def search(self, logical_circuit: Circuit = None, init_mapping: List[int] = None):
        """
        The main process of the qubit mapping algorithm based on the monte carlo tree search. 
        Params:
            logical_circuit: The logical circuit to be transformed into the circuit compliant with the physical device, i.e., 
                            each gate of the transformed circuit are adjacent on the device.
            init_mapping: The initial mapping of the logical quibts to physical qubits.
            coupling_graph: The list of the edges of the physical device's graph.
        """
        self._num_of_executable_gate = 0
        self._logical_circuit_dag = DAG(circuit = logical_circuit, mode = Mode.WHOLE_CIRCUIT) 
        self._circuit_dag = DAG(circuit = logical_circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
        #print(self._logical_circuit_dag.size)
        #print(self._circuit_dag.size)
        self._gate_index = []
        self._physical_circuit: List[BasicGate] = []
        qubit_mask = np.zeros(self._coupling_graph.size, dtype = np.int32) -1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.
        
        
        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit 

        self._root_node = MCTSNode(circuit_dag = self._circuit_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_dag.front_layer, qubit_mask = qubit_mask.copy(), cur_mapping = init_mapping.copy(), parent =None)
       
        
        self._add_initial_single_qubit_gate(node = self._root_node)
        self._add_executable_gates(node = self._root_node)
        
       
        self._expand(node = self._root_node)
        i = 0
        print(self._root_node.num_of_gates)
        while self._root_node.is_terminal_node() is not True:
            i += 1
            t = time.time()
            self._search(root_node = self._root_node)
            self._logger.info("every search time is:%s "%(float)(t-time.time()))
            temp_node = self._root_node
            
            self._root_node = self._decide(node = self._root_node)

            if self._mode == MCTSMode.TRAIN:
                self._transform_to_training_data(node = temp_node, swap_gate = self._root_node.swap_of_edge)
            #print([c.visit_count for c in self._root_node.children ])
            self._logger.info([c.visit_count for c in self._root_node.children ])
            
           
            self._physical_circuit.append(self._coupling_graph.get_swap_gate(self._root_node.swap_of_edge))
            self._add_executable_gates(node = self._root_node)

            print([i,self._num_of_executable_gate])
            self._logger.info([i,self._num_of_executable_gate])
            print(self._root_node.value)
            # print(self._root_node.num_of_gates)
            # print(self._root_node.front_layer)
            # print(self._root_node.sim_value)
            #print(self._root_node.edge_prob)
            # print(self._root_node.reward)
            #print(self._root_node.visit_count)
            #print(self._root_node.candidate_swap_list)
            #print([c.visit_count for c in self._root_node.children ])
            # print(self._num_of_executable_gate )
    # self._gate_index.sort()
    # print(self._gate_index)
        if self._mode == MCTSMode.TRAIN:     
            if self._experience_pool is not None:
                circuit_size = np.array(self._num_list)
                label = np.array(self._label_list)
                adj = np.array(self._adj_list)
                qubits = np.array(self._qubits_list)
                action_probability = np.array(self._action_probability_list)
                value = np.array(self._value_list)
                self._experience_pool.extend(adj = adj, qubits = qubits, action_probability = action_probability, value = value, circuit_size = circuit_size, swap_label = label, num = len(self._adj_list))
            else:
                raise Exception("Experience pool is not defined.")
            
        return self._logical_circuit_dag.size, len(self._physical_circuit)
    

    def _transform_to_training_data(self, node: MCTSNode, swap_gate: int):
        """
        Put the state of the current root node to the experience pool as the tranining data
        """
        num_of_gates = self._size_threshold
        qubit_mask =  [  -1  for _ in range(len(node.cur_mapping))]
        for i, q in  enumerate(node.qubit_mask):
            qubit_mask[node.inverse_mapping[i]] = q

        #print(qubit_mask)
        circuit_size, state = self._circuit_dag.get_subcircuit(front_layer = node.front_layer, qubit_mask = qubit_mask, num_of_gates = num_of_gates)
        #print(1)
        qubit_mapping = np.zeros(len(node.cur_mapping) + 1, dtype = np.int32) -1
        #print(1)
        qubit_mapping[0:-1] = np.array(node.cur_mapping)
        #print(2)
        #adj = np. state[:,0:5]
        adj = np.apply_along_axis(self._fill_adj, axis = 1, arr = np.concatenate([np.arange(0, num_of_gates )[:,np.newaxis], state[0:num_of_gates,0:5]],axis = 1))
        
        qubit_indices = qubit_mapping[state[:,5:]]

        self._label_list.append(swap_gate)
        self._num_list.append(circuit_size)
        self._adj_list.append(adj)
        self._qubits_list.append(qubit_indices)
        self._value_list.append(node.sim_value)
       
        if self._rl == RLMode.WARMUP:
            if self._extended:
                self._action_probability_list.append(node.edge_prob)
            else:
                self._action_probability_list.append(self._extend_action_probability(node.edge_prob, node.candidate_swap_list))
        elif self._rl == RLMode.SELFPALY: 
            if self._extended:
                self._action_probability_list.append(node.visit_count_prob)
            else:
                self._action_probability_list.append(self._extend_action_probability(node.visit_count_prob, node.candidate_swap_list)) 
            

            
    def _search(self, root_node: MCTSNode):
        """
        Monte Carlo tree search from the root node
        """
        #print('search')
        for _ in range(self._selection_times):
            cur_node = self._select(node = root_node)
            self._expand(node = cur_node)
            #aprint(1)
            value = self._rollout(node = cur_node, method = "random")
            self._backpropagate(node = cur_node, value = value)

    def _select(self, node: MCTSNode)-> MCTSNode:
        """
        Select the child node with highest score to expand
        """
        cur_node = node 
        cur_node.visit_count = cur_node.visit_count + 1
        while cur_node.is_leaf_node() is not True:
            cur_node = self._select_next_child(cur_node)
            cur_node.visit_count = cur_node.visit_count + 1
        return cur_node

    def _expand(self, node: MCTSNode):
        """
        Open all child nodes of the current node by applying the swap gates in candidate swap list
        """
        if node is None:
            raise Exception("Node can't be None")
        
        self._get_candidate_swap_list(node = node)

        for swap, prob in zip(node.candidate_swap_list, node.edge_prob):
            node.add_child_node_by_swap_gate(swap, prob)
            
    def _rollout(self, node: MCTSNode, method: str = "random")-> float:
        """
        Do a heuristic search for the sub circuit with Gsim gates from the current node by the specified method
        """
        if self._rs == None:
            if self._sim == SimMode.AVERAGE:
                sim_mode = 0
            elif self._sim == SimMode.MAX:
                sim_mode = 1
            else:
                raise Exception("No such mode")

            self._rs = RandomSimulator(graph = self._circuit_dag.compact_dag, gates = self._circuit_dag.node_qubits,
                                coupling_graph = self._coupling_graph.adj_matrix, distance_matrix = self._coupling_graph.distance_matrix, num_of_swap_gates = 10,
                                num_of_gates = self._circuit_dag.size, num_of_logical_qubits = self._circuit_dag.width, gamma = self._gamma, mode = sim_mode)
        
        res = 0
        front_layer = [ self._circuit_dag.index[i]  for i in node.front_layer ]
        qubit_mask =  [ self._circuit_dag.index[i] if i != -1 else -1 for i in node.qubit_mask ]
            # for i in range(self._Nsim):
            #     N = min(N, self._random_simulation(node))
            #     print("%d and %d"%(N,node.num_of_gates))
        res = self._rs.simulate(front_layer = front_layer, qubit_mapping = node.cur_mapping, qubit_mask = qubit_mask, num_of_subcircuit_gates = self._Gsim, num_of_iterations = self._Nsim ,simulation_mode = 1)
        #print(N)
            #res = np.float_power(self._gamma, N/2) * float(self._Gsim)
            #print(list(front_layer))
            #print(list(qubit_mask))
            #print("%d and %d"%(N,node.num_of_gates))
        node.sim_value = res
        node.value  =   res 
        node.w += res

        return res

    
    def _backpropagate(self, node : MCTSNode, value: int):
        """
        Use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        cur_node = node
        bp_value = self._gamma * value + node.reward
        if self._sim == SimMode.AVERAGE:
            while cur_node.parent is not None:
                cur_node.parent.w += bp_value
                cur_node.parent.value = cur_node.parent.w / cur_node.parent.visit_count
                cur_node = cur_node.parent
                bp_value = cur_node.parent.reward + self._gamma * bp_value
        elif self._sim == SimMode.MAX:
            while cur_node.parent is not None:
                if bp_value > cur_node.parent.value:
                    cur_node.parent.value = bp_value
                bp_value = cur_node.parent.reward + self._gamma * bp_value
                cur_node = cur_node.parent
        else:
            raise Exception("No such mode")



    def _eval(self, node : MCTSNode):
        """
        Evaluate the value of the current node by DNN method
        """
        pass

    def _decide(self,node : MCTSNode)-> MCTSNode:
        """
        Decide which child node to move into 
        """
        node = self._get_best_child(node)
        node.parent = None
        #node.clear()
        return node

    def _select_next_child(self, node: MCTSNode)-> MCTSNode:
        """
        Select the next child to be expanded of the current node 
        """
        res_node = None
        score = float('-inf')
        for child in node.children:
            UCB = self._upper_confidence_bound(node = child)
            if UCB >score:
                res_node = child
                score = UCB
        return res_node
  
    def _get_best_child(self, node: MCTSNode)-> MCTSNode:
        """
        Get the child with highest score of the current node as the next root node. 
        """
        res_node = None
        score = -1
        idx = -1
        # reward_list = []
        # value_list = []
        for i, child in enumerate(node.children):
            # reward_list.append(child.reward)
            # value_list.append(child.value)
            if child.value  > score:
                res_node = child
                score = res_node.value  
                idx = i   
        # print(reward_list)
        # print(value_list) 
        node.best_swap_gate = node.candidate_swap_list[idx]
        return res_node

    def _fall_back(self, node: MCTSNode):
        """
        TODO: If there is still no two-qubit gate can be executed immediately after K consecutive moves, 
              select the gate in the front layer with smallest cost and then make qubits of the gate  adjacent 
              in the physical device by inserting swap gates.  
        """
        pass

    def _add_initial_single_qubit_gate(self, node: MCTSNode):
        """
        Add the single qubit gate in the initial front layer to the physical circuit
        """
        cur_mapping = node.cur_mapping
        sqg_stack = deque(self._logical_circuit_dag.front_layer)
        while len(sqg_stack) > 0:
            gate = sqg_stack.pop()
            if self._logical_circuit_dag[gate]['gate'].is_single():
                self._physical_circuit.append(self._get_physical_gate(gate = self._logical_circuit_dag[gate]['gate'],
                                                cur_mapping = cur_mapping))
                self._gate_index.append(gate)
                for suc in self._logical_circuit_dag.get_successor_nodes(vertex = gate):
                    if self._logical_circuit_dag[suc]['gate'].is_single():
                        sqg_stack.append(suc)
        
    def _add_executable_gates(self, node: MCTSNode):
        """
        Add  executable gates in the node to the physical circuit
        """
        execution_list = node.execution_list
        self._num_of_executable_gate = self._num_of_executable_gate + len(execution_list)
             
        for gate in execution_list:
            if  self._logical_circuit_dag[gate]['gate'].is_single():
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
        cur_mapping = self._root_node.cur_mapping
        sqg_stack = deque([gate])

        mark = set()
        while len(sqg_stack)>0:
            gate_t = sqg_stack.pop()
            self._physical_circuit.append(self._get_physical_gate(gate = self._logical_circuit_dag[gate_t]['gate'],
                                        cur_mapping = cur_mapping))
            # if gate_t in self._gate_index:
            #     raise Exception("The gate index is %d"%(gate))
            self._gate_index.append(gate_t)
            for suc in self._logical_circuit_dag.get_successor_nodes(vertex = gate_t):
                if self._is_valid_successor(suc, gate, gate_t, mark):
                    sqg_stack.append(suc)  
        #print(mark)

    def _is_valid_successor(self, gate_index: int, ori_gate_index: int, pre_gate_index: int, mark: Set[int])->bool:
        """
        Indicate whether the succeeding node can be excuted as soon as the  
        """
        gate = self._logical_circuit_dag[gate_index]['gate']
        pre_gate = self._logical_circuit_dag[pre_gate_index]['gate']
        ori_gate = self._logical_circuit_dag[ori_gate_index]['gate']
        if  gate.is_single():
            return True
        elif gate.controls + gate.targets == 2:
            qubits = self._get_gate_qubits(gate)
            if  ori_gate_index == pre_gate_index:
                pre_qubits = self._get_gate_qubits(pre_gate)
                if is_two_qubit_gate_equal(qubits, pre_qubits):
                    return True
                else:
                    return False
            else:
                ori_qubits = self._get_gate_qubits(ori_gate)
                if is_two_qubit_gate_equal(qubits, ori_qubits):
                    if gate_index in mark:
                            return True
                    else:
                        mark.add(gate_index)
                        return False 
                else:
                    return False
        else:
            raise Exception("The gate type is not supported")


    def _get_gate_qubits(self, gate: BasicGate)->List[int]:
        """
        Get the qubits that the gate acts on 
        """
        if isinstance(gate, BasicGate):
            if gate.is_control_single():
                return [gate.targ, gate.carg]
            elif gate.type() == GATE_ID['Swap']:
                return gate.targs
            else:
                raise Exception("The gate type %d is not supported"%(gate.type()))
        else:
                raise Exception("It is not the gate")



    def test_random_simulation(self, logical_circuit: Circuit, init_mapping: List[int], num_of_gates: int = -1):
        """
        """
        self._num_of_executable_gate = 0
        
        self._logical_circuit_dag = DAG(circuit = logical_circuit, mode = Mode.WHOLE_CIRCUIT) 
        self._circuit_dag = DAG(circuit = logical_circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
        self._gate_index = []
        self._physical_circuit: List[BasicGate] = []
        qubit_mask = np.zeros(self._coupling_graph.size, dtype = np.int32) -1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.
        if num_of_gates == -1:
            self._Gsim = self._circuit_dag.size
        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit 
        
        self._root_node = MCTSNode(circuit_dag = self._circuit_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_dag.front_layer, qubit_mask = qubit_mask.copy(), cur_mapping = init_mapping.copy())

        num_of_swap_gates = self._random_simulation(node = self._root_node)
        #print(num_of_swap_gates)
        if self._mode == MCTSMode.TRAIN:     
            if self._experience_pool is not None:
                circuit_size = np.array(self._num_list)
                label = np.array(self._label_list)
                adj = np.array(self._adj_list)
                qubits = np.array(self._qubits_list)
                action_probability = np.array(self._action_probability_list)
                value = np.array(self._value_list)
                self._experience_pool.extend(adj = adj, qubits = qubits, action_probability = action_probability, value = value, circuit_size = circuit_size, swap_label = label, num = len(self._adj_list))
            else:
                raise Exception("Experience pool is not defined.")
        #print("The average number of inserted swap gates is %s and the time consumed is %s ms"%(num_of_swap_gates, (t2-t1)*1000))

        return num_of_swap_gates, self._circuit_dag.size

    def _random_simulation(self, node: MCTSNode):
        """
        A fast random simulation method for qubit mapping problem. The method randomly select the swap gates 
        from the candidate list by the probability distribution defined by the neareast_neighbour_count of the mapping 
        changed by the swap 
        """
        executed_gate = 0
        cur_node = node.copy()
        self._get_candidate_swap_list(cur_node)
        self._rollout(node = cur_node)

        num_swap = 0
        if self._Gsim == -1:
            threshold = node.num_of_gates
        else:
            threshold = self._Gsim

        while  cur_node.is_terminal_node() is not True and  executed_gate < threshold:    
            list_length = len(cur_node.edge_prob)
            dist_p = cur_node.edge_prob
            #print(dist_p)
            index = np.random.choice(a = list_length, p = dist_p)
            cur_node.swap_of_edge = cur_node.candidate_swap_list[index]
            
            if self._mode == MCTSMode.TRAIN:
                self._transform_to_training_data(node = cur_node, swap_gate = cur_node.swap_of_edge)
            
            cur_node.update_by_swap_gate(swap_index = cur_node.swap_of_edge)
            self._get_candidate_swap_list(cur_node)
            self._rollout(node = cur_node)
            #print(cur_node.value)
       
            executed_gate = executed_gate + len(cur_node.execution_list)
            #print(executed_gate)
            num_swap = num_swap + 1

        
        return num_swap


    def _get_candidate_swap_list(self, node: MCTSNode, pre_prob:bool = False):
        """
        All the swap gates associated with the front layer of the current node
        """
        qubits_set = self._get_invloved_qubits(node = node)
        candidate_swap_set = set()

        for qubit in qubits_set:
            for adj in self._coupling_graph.get_adjacent_vertex(qubit):
                candidate_swap_set.add(frozenset([qubit,adj]))
                
        candidate_swap_list = []
        for swap_index_set in candidate_swap_set:
            swap_index_list = list(swap_index_set)
            if  node.parent is None or node.parent.swap_of_edge != -1 or not is_two_qubit_gate_equal(swap_index_list, self._coupling_graph.get_swap_gate(node.parent.swap_of_edge).targs):
                GateBuilder.setGateType(GATE_ID['Swap']) 
                GateBuilder.setTargs(swap_index_list)
                candidate_swap_list.append(self._coupling_graph.edge_label(GateBuilder.getGate()))
        
        node.candidate_swap_list =  candidate_swap_list
        
        if pre_prob is not True:
            edge_prob_cal = EdgeProb(circuit = self._circuit_dag, coupling_graph = self._coupling_graph, qubit_mapping = node.cur_mapping, gates = node.front_layer)
            base =  edge_prob_cal()
            edge_prob = f(np.array([base - edge_prob_cal(swap) for swap in node.candidate_swap_list ], dtype=np.float))
            edge_prob = edge_prob / np.sum(edge_prob)
            node.edge_prob = edge_prob


    def _extend_action_probability(self, action_probability: np.ndarray, swap_edges: List[int]):
        res = np.zeros(self._coupling_graph.num_of_edge, dtype = float)
        for p, idx in zip(action_probability, swap_edges):  
            res[idx] = p
        return res


    def _get_invloved_qubits(self, node: MCTSNode)-> List[int]:
        """
        Get the list of the physical qubits dominated by the gates in front layer 
        """
        qubit_set = set()
        for gate in node.front_layer:
            if self._gate_qubits(gate) == 1:
                qubit_set.add(self._get_gate_target(gate, 0, cur_mapping = node.cur_mapping))
            elif self._is_swap(gate) is not True:
                qubit_set.add(self._get_gate_control(gate, 0, cur_mapping = node.cur_mapping))
                qubit_set.add(self._get_gate_target(gate, 0, cur_mapping = node.cur_mapping))
            else:
                qubit_set.add(self._get_gate_target(gate,0, cur_mapping = node.cur_mapping))
                qubit_set.add(self._get_gate_target(gate,1, cur_mapping = node.cur_mapping))
        return list(qubit_set)

    def _gate_qubits(self, gate_in_dag: int)->int:
        """
        The number of the qubits dominated by the gate.
        """            
        return self._circuit_dag[gate_in_dag]['gate'].controls + self._circuit_dag[gate_in_dag]['gate'].targets

    def _get_gate_control(self, gate_in_dag: int, index: int = 0, cur_mapping: List[int] = None)->int:
        """
        The physical qubit of the gate 'index'-th control qubit under the current mapping 
        """
        if  index < self._circuit_dag[gate_in_dag]['gate'].controls:
            return cur_mapping[self._circuit_dag[gate_in_dag]['gate'].cargs[index]]
        else:
            raise  IndexLimitException(self._circuit_dag[gate_in_dag]['gate'].controls, index)

    def _get_gate_target(self, gate_in_dag: int, index: int = 0, cur_mapping: List[int] = None)->int:  
        """
        The physical qubit of the gate 'index'-th target qubit under the current mapping 
        """
        return cur_mapping[self._circuit_dag[gate_in_dag]['gate'].targs[index]]
    
    def _is_swap(self, gate_in_dag: int)->bool:
        """
        Indicate whether the gate is a swap gate
        """
        return self._circuit_dag[gate_in_dag]['gate'].type() == GATE_ID['Swap']

        

        
        
    
    
    




    


