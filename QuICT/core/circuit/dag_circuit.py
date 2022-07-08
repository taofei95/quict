import networkx as nx
import matplotlib.pyplot as plt

from QuICT.core.gate import BasicGate


class DAGNode:
    @property
    def id(self):
        return self._id

    @property
    def gate(self):
        return self._gate

    def __init__(self, id: int, gate: BasicGate):
        self._id = id
        self._gate = gate


class DAGCircuit:
    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return self._size

    @property
    def width(self) -> int:
        return self._width

    def __init__(self, circuit):
        self._circuit = circuit
        self._name = f"DAG_{self._circuit.name}"
        self._size = self._circuit.size()
        self._width = self._circuit.width()
        self._graph = nx.DiGraph()
        self._to_dag_circuit()

    ####################################################################
    ############         DAGCircuit Nodes and Edges         ############
    ####################################################################
    def __getitem__(self, item):
        """ to fit the slice operator, overloaded this function.

        get a smaller qureg/qubit from this circuit

        Args:
            item(int/slice): slice passed in.
        Return:
            Qubit/Qureg: the result or slice
        """
        return self.nodes()[item]

    def nodes(self) -> list:
        return list(self._graph.nodes)

    def add_node(self, node: DAGNode):
        assert isinstance(node, DAGNode)
        if not self._graph.has_node(node.id):
            self._graph.add_node(node.id, node=node)

    def get_node(self, node_id: int):
        return self._graph.nodes[node_id]["node"]

    def edges(self) -> list:
        return list(self._graph.edges)

    def in_edges(self, node_id):
        return self._graph.in_edges(node_id)
    
    def out_edges(self, node_id):
        return self._graph.out_edges(node_id)

    ####################################################################
    ############                Circuit to DAG              ############
    ####################################################################
    def _to_dag_circuit(self):
        gates = self._circuit.gates
        endpoints = []      # The endpoint of current DAG graph
        for idx, gate in enumerate(gates):
            current_edges = self._graph.number_of_edges()
            # Add new node into DAG Graph
            assert isinstance(gate, BasicGate), "Only support BasicGate in DAGCircuit."
            current_node = DAGNode(idx, gate)
            self.add_node(current_node)

            # Find predecessors for current node
            # TODO: _backward_trace should deal with by endpoint's layer
            updated_endpoints, required_trace = [], True
            for previous_node in endpoints:
                is_matched = self._backward_trace(previous_node, current_node, required_trace)
                if is_matched:
                    updated_endpoints.append(current_node)
                    if not self._graph.has_edge(previous_node.id, idx):
                        updated_endpoints.append(previous_node)

                    required_trace = False
                else:
                    updated_endpoints.append(previous_node)

            # if no edges add, create new original node
            if current_edges == self._graph.number_of_edges():
                endpoints.insert(0, current_node)
            else:
                endpoints = self._endpoints_order(updated_endpoints)

    def _backward_trace(self, previous_node, current_node, trace: bool = True):
        cgate = current_node.gate
        point = [previous_node]
        matched = False
        while len(point) != 0:
            pgate = point.pop()
            if isinstance(pgate, int):
                pgate = self.get_node(pgate)

            if not pgate.gate.commutative(cgate):
                self._graph.add_edge(pgate.id, current_node.id)
                matched = True
                break

            if trace:
                pred_list = list(self._graph.predecessors(pgate.id))
                point += pred_list

        return matched

    def _endpoints_order(self, endpoints: list):
        node_ids = [endpoint.id for endpoint in endpoints]
        # remove duplicated item and sort by decreasing
        node_ids = list(set(node_ids))
        node_ids.sort(reverse=True)

        return [self.get_node(nid) for nid in node_ids]        

    ####################################################################
    ############              DAG_Circuit utils             ############
    ####################################################################
    def draw(self, layout=nx.shell_layout):
        graph_name = f"{self.name}.jpg"
        nx.draw(self._graph, pos=layout(self._graph), with_labels=True)
        plt.savefig(graph_name)
