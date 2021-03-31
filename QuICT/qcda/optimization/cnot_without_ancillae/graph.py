import copy


class Edge:
    """
    A simple container for edges in a graph

    Attribute
    ----------
    start : int
            start point of the edge
    end : int
        end point of the edge
    valid : bool
        attribute of the edge indicating whether it's valid
    color : int
        color of the edge
    next : int
        next fake pointer in edge list
    """

    def __init__(self, start: int, end: int, valid: bool = True, color: int = -1, _next: int = -1):
        """
        Construct an edge
        """
        self.start = start
        self.end = end
        self.valid = valid
        self.color = color
        self.next = _next
        """next edge(which is used in head array)"""

    def __repr__(self):
        return f"edge [start: {self.start}," \
               f" end: {self.end}, " \
               f"valid: {self.valid}, " \
               f"color: {self.color}, " \
               f"next={self.next}]"


class Graph:
    """
    A simple container for graphs. The graph is directed.

    For undirected graph, you must add reverse edges just after
    adding an edge. For example, add_edge(a,b), add_edge(b, a).
    Therefore, `eid` of reverse edge is exactly `eid^1`.
    """

    def __init__(self, nodes=None, edges=None):
        """
        Construct a graph. It's empty if both nodes and edges are None.

        Parameters
        ----------
        nodes : List
            A list of all nodes. In most cases, it should be a continuous list
            which contains sequential integers.
        edges : List
            A list of all edges
        """
        self.nodes = [] if nodes is None else copy.copy(nodes)
        self.edges = []
        self.head = {}
        for i in self.nodes:
            self.head[i] = -1
        """get edges that share the same head(start point)"""
        if edges is None:
            return
        cnt = 0
        for edge in edges:
            s = edge.start
            e = edge.end
            m = edge.valid
            self.edges.append(Edge(s, e, m, self.head[s]))
            self.head[s] = cnt
            cnt += 1

    def get_degree(self, x: int):
        """get degree of node x"""
        deg = 0
        i = self.head[x]
        while i != -1:
            deg += 1
            i = self.edges[i].next
        return deg

    def get_any_degree(self) -> int:
        return self.get_degree(self.nodes[0])

    def get_max_degree(self) -> int:
        """get maximum degree of the graph

        Returns
        -------
        int
            max degree
        """
        deg = -1
        for i in self.nodes:
            deg = max(deg, self.get_degree(i))
        return deg

    def get_max_degree_node(self) -> int:
        index = -1
        deg = -1
        for node in self.nodes:
            new_deg = self.get_degree(node)
            if deg < new_deg:
                deg = new_deg
                index = node
        return index

    def list_degree(self) -> None:
        s = 0
        for node in self.nodes:
            d = self.get_degree(node)
            print(d)
            s += d
        print(f"summation of degree: {s}")

    def add_edge(self, start: int, end: int, valid: bool = True, color: int = -1):
        """add an edge from start end with marked value

        Parameters
        ----------
        start : int
            start point
        end : int
            end point
        valid : bool
            if valid
        color : int
            color of the edge

        Returns
        -------
        None
        """
        self.edges.append(Edge(start=start, end=end, valid=valid, color=color, _next=self.head[start]))
        self.head[start] = len(self.edges) - 1


class Bipartite(Graph):
    """A bipartite graph

    Attribute
    ---------
    left : array-like
        left side of bipartite graph
    right : array-like
        right side of bipartite graph
    """

    def __init__(self, left, right, edges=None):
        """Construct a bipartite graph. There should be no overlap between left and right

        Parameters
        ----------
        left : array-like
            left side of bipartite graph
        right : array-like
            right side of bipartite graph
        edges : array-like
            Specify edge between left and right. The next field of Edge could be invalid here.
        """
        self.left = copy.copy(left)
        self.right = copy.copy(right)
        super().__init__(left + right, edges)

    def list_degree(self) -> None:
        s = 0
        print("left side:")
        for node in self.left:
            d = self.get_degree(node)
            print(d)
            s += d
        print("right side:")
        for node in self.right:
            d = self.get_degree(node)
            print(d)
            s += d
        print(f"summation of degree: {s}")
