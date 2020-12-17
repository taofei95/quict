from  ._mcts_base import  *

class TableBasedMCTS(MCTSBase):
    def __init__(self, play_times: int =20, mode: str = None, mode_sim :List[int] = None,
                selection_times: int = 50, coupling_graph : List[Tuple] = None,  **params):
        super.__init__(coupling_graph)
        self.play_times = play_times 
        self.mode = mode
        self.mode_sim = mode_sim
        self.selection_times = selection_times 
        self.call_back_threshold = 100
    
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
    
    def search(self, circuit: Circuit, layout: List[int]):
        self._logical_circuit = DAG(circuit = circuit) 
        self._physical_circuit =Circuit(circuit.wires)
        self._root_node = MCTSNode(circuit = self._logical_circuit , coupling_graph = self._coupling_graph, front_layer = self._logical_circuit.front_layer,
                                  layout = layout)
        
        self._add_excutable_gates(self.root_node)
        self._expand(node = self._root_node)
        while self._root_node.is_terminal_node():
            self._search(root_node = self._root_node)
            self._root_node, swap_gate = self._decide(node = self._root_node)
            self._physical_circuit.append(swap_gate)

    def _search(self, root_node : MCTSNode):
        for i in range(self.selection_times):
            cur_node = self._select(node = root_node)
            self._expand(node = cur_node)
            self._rollout(node = cur_node, method = "random")
            self._backpropagate(cur_node)

    def _select(self, node : MCTSNode):
        """
        select the child node with highes score to expand
        """
        cur_node = node 
        while cur_node.is_leaf_node() is not True:
            cur_node = self._get_best_child()
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
        if method == "random":
            pass

    
    def _backpropagate(self,node : MCTSNode):
        """
        use the result of the rollout to update the score in the nodes on the path from the current node to the root 
        """
        cur_node = node
        while cur_node.parent is not None:
            cur_node.update_node()
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
        return node.decide_best_child() 

    def _get_best_child(self, node: MCTSNode):
        """
        TODO: move code here instead of MCTSNode
        """
        return node.get_best_child(node)

    def _call_back(self, node: MCTSNode):
        """

        """
        pass

    def _add_excutable_gates(self, node: MCTSNode):
        excution_list = self._root_node.excution_list()
        self._physical_circuit.extend(excution_list)

    def _random_simulation(self, node: MCTSNode, ):
        pass
        
    
    
    


        


    


