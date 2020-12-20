from  ._mcts_base import  *
from queue import Queue, deque



class TableBasedMCTS(MCTSBase):
    def __init__(self, play_times: int =20, mode: str = None, mode_sim :List[int] = None,
                selection_times: int = 50, coupling_graph : List[Tuple] = None,  **params):
        super.__init__(coupling_graph)
        # TODO
        self._play_times = play_times 
        self._mode = mode
        self._mode_sim = mode_sim
        self._selection_times = selection_times 
        self._call_back_threshold = 100
        self._gamma = 0.6
        self._Gsim = 10
        self._Nsim = 5
        self._coupling_graph = coupling_graph
    
    @property
    def physical_circuit(self):
        """
        the physical circuit of the current root node
        """
        return self._root_node.circuit

    @property
    def root_node(self):
        """

        """
        return self._root_node


    #TODO: layout -> mapping
    def search(self, logical_circuit: Circuit, cur_mapping: List[int]):
        self._circuit_in_dag = DAG(circuit = logical_circuit) 
        self._physical_circuit =Circuit(logical_circuit.circuit_length())
        self._root_node = MCTSNode(circuit_in_dag = self._circuit_in_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_in_dag.front_layer, cur_mapping = cur_mapping)
        
        self._add_excutable_gates(self.root_node)
        self._expand(node = self._root_node)
        while self._root_node.is_terminal_node():
            self._search(root_node = self._root_node)
            self._root_node, swap_gate = self._decide(node = self._root_node)
            self._physical_circuit.append(swap_gate)

    def _search(self, root_node : MCTSNode):
        for i in range(self._selection_times):
            cur_node, swap = self._select(node = root_node)
            self._expand(node = cur_node)
            self._rollout(node = cur_node, method = "random")
            self._backpropagate(cur_node)

    def _select(self, node : MCTSNode)-> Tuple[MCTSNode, SwapGate]:
        """
        select the child node with highes score to expand
        """
        cur_node = node 
        swap = SwapGate()
        while cur_node.is_leaf_node() is not True:
            cur_node, swap = self._select_next_child(cur_node)
            cur_node.visit_count =  cur_node.visit_count + 1
        return cur_node, swap

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
        if method == "random":
            for i in range(self._Nsim):
               N = min(N, self._random_simulation(node))
            res = np.float_power(self._gamma, N/2) * float(self._Gsim)
        node.value = res

    
    def _backpropagate(self,node : MCTSNode):
        """
        use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        cur_node = node
        while cur_node.parent is not None:
            if self._gamma*(cur_node.value + cur_node.reward) > cur_node.parent.vlaue:
                cur_node.parent.vlaue = self._gamma*(cur_node.value + cur_node.reward)
                cur_node = cur_node._parent
            else:
                break



    def _eval(self, node : MCTSNode):
        """
        evaluate the vlaue of the current node by DNN method
        """
        pass

    def _decide(self,node : MCTSNode):
        """
        decide which child node to move into 
        """
        return node._get_best_child() 

    #TODO: _select_next_child
    def _select_next_child(self, node: MCTSNode)-> Tuple[MCTSNode, SwapGate]:
        """
        """
        res_node = MCTSNode()
        res_swap =  BasicGate()
        score = -1
        for child in node.children:
            if child[0].upper_confidence_bound >score:
                res_node = child[0]
                res_swap = child[1]
                score = res_node.upper_confidence_bound
        return res_node, res_swap


        
    def _get_best_child(self, node: MCTSNode):
        """
        TODO: move code here instead of MCTSNode
        """
        res_node = MCTSNode()
        res_swap =  BasicGate()
        score = -1
        for child in node.children:
            if child[0].value + child[0].reward >score:
                res_node = child[0]
                res_swap = child[1]
                score = res_node.value + res_node.reward  

        return res_node, res_swap

    def _call_back(self, node: MCTSNode):
        """

        """
        pass

    def _add_excutable_gates(self, node: MCTSNode):
        excution_list = self._root_node.excution_list()
        self._physical_circuit.extend(excution_list)

    def _gate_distance_in_device(self, cur_mapping: List[int])->Callable[[int],int]:
        """
        """
        cur_mapping = cur_mapping
        def func(gate_in_dag: int)->int:
            return  self._coupling_graph.distance(cur_mapping[self._circuit_in_dag[gate_in_dag]['gate'].carg], cur_mapping[self._circuit_in_dag[gate_in_dag]['gate'].targ])
        return func

    def _neareast_neighbour_count(self, front_layer: List[int], cur_mapping: List[int] )-> int:
        """
        """
        gate_distance = self._gate_distance_in_device(cur_mapping = cur_mapping)
        fl_queue = deque(front_layer)
        mark_set = set(fl_queue)
        NNC = 0
        while len(fl_queue) != 0:
            gate = fl_queue.popleft()
            mark_set.remove(gate)
            NNC = NNC + gate_distance(gate)
            for suc in self._circuit_in_dag.get_successor_nodes(gate):
                if suc in mark_set:
                    fl_queue.append(suc)
                    mark_set.add(suc)
        return NNC

    def _change_mapping_with_single_swap(self, cur_mapping: List[int], swap_gate: SwapGate)->List[int]:
        """
        
        """
        res_mapping = cur_mapping.copy()

        if isinstance(swap_gate, SwapGate):
            l_control = swap_gate.carg
            l_target = swap_gate.targ
            p_control = cur_mapping[l_control]
            p_target  = cur_mapping[l_target]
            if self._coupling_graph.is_adjacent(p_control, p_target):
                res_mapping[l_control] = p_target
                res_mapping[l_target] = p_control
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate","other gate")

        return res_mapping

    def _f(self, x: Optional[np.ndarray, int])->float:
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
        cur_node = copy.copy(node)
        num_swap = 0
        while excuted_gate < self._Gsim:
            base = self._neareast_neighbour_count(front_layer = cur_node.front_layer, cur_mapping = cur_node.cur_mapping)
            list_length = len(cur_node.candidate_swap_list)
            NNC= np.zeros(list_length, dtype=float)

            for i,swap_gate in enumerate(cur_node.candidate_swap_list):
                mapping = self._change_mapping_with_single_swap(cur_node.cur_mapping, swap_gate)
                NNC[i] = self._neareast_neighbour_count(cur_mapping = mapping, front_layer = cur_node.front_layer) - base
           
            dist_p = self._f(NNC)/np.sum(NNC)
            index = np.random.choice(a = list_length, p = dist_p)

            mapping = self._change_mapping_with_single_swap(cur_node.cur_mapping, cur_node.candidate_swap_list[index])
            cur_node.cur_mapping = mapping
            cur_node.update_node()

            excuted_gate = excuted_gate + len(cur_node.excution_list)
            num_swap = num_swap + 1
        return num_swap
            

        

        
        
    
    
    


        


    


