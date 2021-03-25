import time
from queue import Queue, deque

from  mcts_cpp.mcts_wrapper  import MCTSTreeWrapper
from  mcts_node import  *


class MCTS(object):
    @classmethod
    def _get_physical_gate(cls, gate: BasicGate, cur_mapping: List[int])->BasicGate:
        """
        Get the  physical gate of the given logical gate   
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
   
    def __init__(self, play_times: int = 1, gamma: float = 0.7, Gsim: int = 30,
                 Nsim: int = 500, selection_times: int = 40, c = 20, graph_name: str = None, **params):
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
        self._call_back_threshold = 100
        self._gamma = gamma
        self._Gsim = Gsim
        self._Nsim = Nsim
        self._c = c
        self._coupling_graph = get_coupling_graph(graph_name = graph_name)
        self._mcts_wrapper = MCTSTreeWrapper(gamma = self._gamma, c = self._c, size_of_subcircuits = self._Gsim, num_of_iterations = self._selection_times, num_of_playout = self._Nsim, num_of_edges = self._coupling_graph.num_of_edge, 
                                        coupling_graph = self._coupling_graph.adj_matrix, distance_matrix = self._coupling_graph.distance_matrix, edge_label = self._coupling_graph.label_matrix, feature_matrix = self._coupling_graph.node_feature) 
    
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
        self._logical_circuit_dag = DAG(circuit = logical_circuit, mode = Mode.WHOLE_CIRCUIT) 
        self._circuit_dag = DAG(circuit = logical_circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
        self._gate_index = []
        self._physical_circuit: List[BasicGate] = []
        qubit_mask = np.zeros(self._coupling_graph.size, dtype = np.int32) -1
        # For the logical circuit, its initial mapping is [0,1,2,...,n] in default. Thus, the qubit_mask of initial logical circuit
        # should be mapping to physical device with the actual initial mapping.
        
        
        for i, qubit in enumerate(self._circuit_dag.initial_qubit_mask):
            qubit_mask[init_mapping[i]] = qubit 
        
        front_layer_ = [ self._circuit_dag.index[i]  for i in self._circuit_dag.front_layer ]
        qubit_mask_ =  [ self._circuit_dag.index[i] if i != -1 else -1 for i in qubit_mask ]

        self._mcts_wrapper.load_data(num_of_logical_qubits = logical_circuit.circuit_width(), circuit = self._circuit_dag.node_qubits, dependency_graph = self._circuit_dag.compact_dag,
                                 qubit_mapping = init_mapping, qubit_mask = qubit_mask_, front_layer =  front_layer_)

        self._cur_node = MCTSNode(circuit_dag = self._circuit_dag , coupling_graph = self._coupling_graph,
                                  front_layer = self._circuit_dag.front_layer, qubit_mask = qubit_mask.copy(), cur_mapping = init_mapping.copy())
        
        self._add_initial_single_qubit_gate(node = self._cur_node)
        self._add_executable_gates(node = self._cur_node)

        res =  self._mcts_wrapper.search()
        print(res)
        

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
        cur_mapping = self._cur_node.cur_mapping
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
        Indicate whether the succeeding node can be excuted as soon as the  
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
        Get the qubits that the gate acts on 
        """
        if gate.is_control_single():
            return [gate.targ, gate.carg]
        elif gate.type() == GATE_ID['Swap']:
            return gate.targs
        else:
            raise Exception("The gate type %d is not supported"%(gate.type()))
