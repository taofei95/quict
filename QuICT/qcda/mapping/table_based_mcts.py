from  _mcts_base import  *
from queue import Queue, deque


class TableBasedMCTS(MCTSBase):
    
    @classmethod
    def _get_physical_gate(cls, gate: BasicGate, cur_mapping: List[int])->BasicGate:
        """
        """
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
        return cur_gate
   
    def __init__(self, play_times: int = 20, gamma: float = 0.7, Gsim: int = 30,
                 Nsim: int = 5, selection_times: int = 20, c = 20, **params):
        # TODO
        self._play_times = play_times 

        self._selection_times = selection_times 
        self._call_back_threshold = 100
        self._gamma = gamma
        self._Gsim = Gsim
        self._Nsim = Nsim
        self._c = c
    
    @property
    def physical_circuit(self):
        """
        the physical circuit of the current root node
        """
        return self._physical_circuit

    @property
    def root_node(self):
        """

        """
        return self._root_node
    
    def _upper_confidence_bound(self, node: MCTSNode)->float:
        """
        The upper confidence bound of the node
        """
        return node.reward + node.value + self._c* np.sqrt(np.log2(node.parent.visit_count) / node.visit_count)
    

    #TODO: layout -> mapping
    def search(self, logical_circuit: Circuit, cur_mapping: List[int], coupling_graph : List[Tuple] = None):
        """

        """
        
        self._num_of_excutable_gate = 0
        self._coupling_graph = CouplingGraph(coupling_graph = coupling_graph)
        self._logical_circuit_dag = DAG(circuit = logical_circuit, mode = 1) 
        self._circuit_dag = DAG(circuit = logical_circuit, mode = 2)
        #self._circuit_dag.draw()
        #print(self._circuit_dag.size)
        #self._physical_circuit = Circuit(logical_circuit.circuit_length())
        self._gate_index = []
        self._physical_circuit: List[BasicGate] = []
        #print(self._circuit_dag.front_layer)
        qubit_mask = np.zeros(self._coupling_graph.size, dtype = np.int32) -1
        for i, qubit in enumerate(self._circuit_dag.qubit_mask):
            qubit_mask[cur_mapping[i]] = qubit 
        
        self._root_node = MCTSNode(circuit_dag = self._circuit_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_dag.front_layer, qubit_mask = qubit_mask, cur_mapping = cur_mapping)
        
        self._add_initial_single_qubit_gate(self._root_node)
        self._add_excutable_gates(self._root_node)
        self._expand(node = self._root_node)
        i=0
        while self._root_node.is_terminal_node() is not True:
            i = i+1
            # print(self._root_node.front_layer)
            # print(self._root_node.reward)
            self._search(root_node = self._root_node)
            # for child in self._root_node.children:
            #     print(child.front_layer)
            #     print(child._qubit_mask)
            #     print(child.reward)
            #     print(child.value)
            #print(self._root_node.front_layer)
            #print(self._root_node.cur_mapping)
            self._root_node = self._decide(node = self._root_node)
            #print(self._root_node.front_layer)
            self._physical_circuit.append(self._root_node.swap_with_edge)
            self._add_excutable_gates(self._root_node)
            #print("The %d-th search: the current size of physical circuit:%d"%(i, self._num_of_excutable_gate))
            # self._gate_index.sort()
            # print(self._gate_index)


    def _search(self, root_node: MCTSNode):
        for i in range(self._selection_times):
            #print("MCTS:%d"%(i))
            cur_node = self._select(node = root_node)
            self._expand(node = cur_node)
            self._rollout(node = cur_node, method = "random")
            self._backpropagate(cur_node)

    def _select(self, node: MCTSNode)-> MCTSNode:
        """
        select the child node with highes score to expand
        """
        cur_node = node 
        cur_node.visit_count = cur_node.visit_count + 1
        while cur_node.is_leaf_node() is not True:
            cur_node = self._select_next_child(cur_node)
            cur_node.visit_count = cur_node.visit_count + 1
        return cur_node

    def _expand(self, node : MCTSNode):
        """
        open all child nodes of the  current node by applying the swap gates in candidate swap list
        """
        for swap in node.candidate_swap_list:
            node.add_child_node(swap)
            

    def _rollout(self, node : MCTSNode, method: str):
        """
        complete a heuristic search from the current node
        """
        res = 0
        N = np.inf
        if method == "random":
            for i in range(self._Nsim):
              # print("   roll out:%d"%(i))
               N = min(N, self._random_simulation(node))
            #print(N)
            res = np.float_power(self._gamma, N/2) * float(self._Gsim)
        node.value = res
        #print(res)

    
    def _backpropagate(self,node : MCTSNode):
        """
        use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        cur_node = node
        while cur_node.parent is not None:
            if self._gamma*(cur_node.value + cur_node.reward) > cur_node.parent.value:
                cur_node.parent.vlaue = self._gamma*(cur_node.value + cur_node.reward)
            cur_node = cur_node._parent


    def _eval(self, node : MCTSNode):
        """
        evaluate the vlaue of the current node by DNN method
        """
        pass

    def _decide(self,node : MCTSNode):
        """
        decide which child node to move into 
        """
        return self._get_best_child(node) 

    #TODO: _select_next_child
    def _select_next_child(self, node: MCTSNode)-> MCTSNode:
        """
        """
        res_node = MCTSNode()
        score = -1
        for child in node.children:
            UCB = self._upper_confidence_bound(node = child)
            #UCB = child.upper_confidence_bound
            if UCB >score:
                res_node = child
                score = UCB
        return res_node
  
    def _get_best_child(self, node: MCTSNode)-> MCTSNode:
        """
        TODO: move code here instead of MCTSNode
        """
        res_node = MCTSNode()
        score = -1
        for child in node.children:
            if child.value + child.reward >score:
                res_node = child
                score = res_node.value + res_node.reward  
        return res_node

    def _call_back(self, node: MCTSNode):
        """

        """
        pass

    def _add_initial_single_qubit_gate(self, node: MCTSNode):
        """
        add the single qubit gate in the initial front layer to the physical circuit
        """
        cur_mapping = self._root_node.cur_mapping
        sqg_stack = deque(self._logical_circuit_dag.front_layer)
        while len(sqg_stack) > 0:
            gate = sqg_stack.pop()
            if self._logical_circuit_dag[gate]['gate'].is_single():
                self._physical_circuit.append(self._get_physical_gate(gate = self._logical_circuit_dag[gate]['gate'],
                                                cur_mapping = cur_mapping))
                for suc in self._logical_circuit_dag.get_successor_nodes(vertex = gate):
                    if self._logical_circuit_dag[suc]['gate'].is_single():
                        sqg_stack.append(suc)
        
    def _add_excutable_gates(self, node: MCTSNode):
        """
        add  excutable gates in the node to the physical circuit
        """
        excution_list = self._root_node.excution_list
        self._num_of_excutable_gate = self._num_of_excutable_gate + len(excution_list)
        # self._gate_index.extend(self._root_node._gate_index)
        
            
        for gate in excution_list:
            if  self._logical_circuit_dag[gate]['gate'].is_single():
                raise Exception("There shouldn't exist single qubit gate in two-qubit gate circuit")
            else:
                self._add_gate_and_successors(gate)

    
    def _add_gate_and_successors(self, gate: int):
        """
        add the current gate and  single qubit gates  succeed it to physical circuit 
        """
        cur_mapping = self._root_node.cur_mapping
        sqg_stack = deque([gate])

        mark = set()
        while len(sqg_stack)>0:
            gate_t = sqg_stack.pop()
            self._physical_circuit.append(self._get_physical_gate(gate = self._logical_circuit_dag[gate_t]['gate'],
                                        cur_mapping = cur_mapping))
            for suc in self._logical_circuit_dag.get_successor_nodes(vertex = gate_t):
                if self._is_valid_successor(suc, gate, mark):
                    sqg_stack.append(suc)  

    def _is_valid_successor(self, gate_index: int, pre_gate_index: int, mark: Set[int])->bool:
        """
        """
        gate = self._logical_circuit_dag[gate_index]['gate']
        pre_gate = self._logical_circuit_dag[pre_gate_index]['gate']
        if gate.is_single():
            return True
        elif gate.controls + gate.targets == 2:
            qubits = self._get_gate_qubits(gate)
            pre_qubits = self._get_gate_qubits(pre_gate)
            if gate not in mark and (qubits[0] == pre_qubits[0] and qubits[1] == pre_qubits[1]) or (qubits[0] == pre_qubits[1] and qubits[1] == pre_qubits[0]):
               mark.add(gate)
               return True
            else:
                False 
        else:
            raise Exception("The gate type is not supported")

    def _get_gate_qubits(self, gate: BasicGate)->List[int]:
        """
        return the qubits that the gate acts on 
        """
        if gate.is_control_single():
            return [gate.targ, gate.carg]
        elif gate.type() == GATE_ID['Swap']:
            return gate.targs
        else:
            raise Exception("The gate type %d is not supported"%(gate.type()))

    def _gate_distance_in_device(self, cur_mapping: List[int])->Callable[[int],int]:
        """
        return a function that can calculate the distance on the device between the control qubit and target qubit of the given gate 
        """
        cur_mapping = cur_mapping
        def func(gate_in_dag: int)->int:
            return  self._coupling_graph.distance(cur_mapping[self._circuit_dag[gate_in_dag]['gate'].carg], cur_mapping[self._circuit_dag[gate_in_dag]['gate'].targ])
        return func

    def _neareast_neighbour_count(self, front_layer: List[int], cur_mapping: List[int] )-> int:
        """
        """
        gate_distance = self._gate_distance_in_device(cur_mapping = cur_mapping)
        NNC = 0
        for gate in front_layer:
            NNC = NNC + gate_distance(gate)
        return NNC

    def _change_mapping_with_single_swap(self, cur_mapping: List[int], swap_gate: SwapGate)->List[int]:
        """
        
        """
        res_mapping = cur_mapping.copy()

        if isinstance(swap_gate, SwapGate):
            p_target = swap_gate.targs
            l_target = [cur_mapping.index(p_target[i], 0, len(cur_mapping)) 
                        for i in  range(len(p_target)) ]
    
            if self._coupling_graph.is_adjacent(p_target[0], p_target[1]):
                res_mapping[l_target[0]] = p_target[1]
                res_mapping[l_target[1]] = p_target[0]
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate","other gate")

        return res_mapping
    def _change_qubit_mask_with_single_swap(self, qubit_mask: List[int], swap_gate: SwapGate)->List[int]:
        """
        
        """
        res_qubit_mask = qubit_mask.copy()

        if isinstance(swap_gate, SwapGate):
            p_target = swap_gate.targs
    
            if self._coupling_graph.is_adjacent(p_target[0], p_target[1]):
                res_qubit_mask[p_target[0]] = qubit_mask[p_target[1]]
                res_qubit_mask[p_target[1]] = qubit_mask[p_target[0]]
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate","other gate")

        return res_qubit_mask

    def _f(self, x: np.ndarray = None)->float:
        if isinstance(x, int):
            if x < 0:
                return 0
            elif x == 0:
                return 0.001
            else:
                return x
        else :
            return np.piecewise(x, [x<0, x==0, x>0], [0, 0.001, lambda x: x])

    def _random_simulation(self, node: MCTSNode):
        excuted_gate = 0
        cur_node = node.copy()
        num_swap = 0

        while  cur_node.is_terminal_node() is not True and  excuted_gate < self._Gsim:
            # if len(cur_node.front_layer) != 0 and len(cur_node.candidate_swap_list)==0:
            #     print(1)
            base = self._neareast_neighbour_count(front_layer = cur_node.front_layer, cur_mapping = cur_node.cur_mapping)
            
            list_length = len(cur_node.candidate_swap_list)
            NNC= np.zeros(list_length, dtype=float)

            for i,swap_gate in enumerate(cur_node.candidate_swap_list):
                mapping = self._change_mapping_with_single_swap(cur_node.cur_mapping, swap_gate)
                NNC[i] = base - self._neareast_neighbour_count(cur_mapping = mapping, front_layer = cur_node.front_layer)
            #print(NNC)
            dist_p = self._f(NNC)
            dist_p = dist_p / np.sum(dist_p)
            index = np.random.choice(a = list_length, p = dist_p)
            #print(dist_p)

            cur_node.cur_mapping  = self._change_mapping_with_single_swap(cur_node.cur_mapping, cur_node.candidate_swap_list[index])
            cur_node.qubit_mask = self._change_qubit_mask_with_single_swap(cur_node.qubit_mask, cur_node.candidate_swap_list[index])
            # if num_swap % 100 == 0:
            #     print(mapping)
            #cur_node.cur_mapping = mapping
            cur_node.swap_with_edge = cur_node.candidate_swap_list[index].copy()

            cur_node.update_node()

            excuted_gate = excuted_gate + len(cur_node.excution_list)
            #print("excuted gate:%d"%excuted_gate)
            num_swap = num_swap + 1
            #print(excuted_gate)

            

        return num_swap
            

        

        
        
    
    
    


        


    


