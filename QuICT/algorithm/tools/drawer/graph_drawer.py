"""A collection of utility functions for drawing."""


import matplotlib.pyplot as plt
import networkx as nx


OPTIONS = {
    "with_labels": True,
    "font_size": 20,
    "font_weight": "bold",
    "font_color": "white",
    "node_size": 2000,
    "width": 2,
}


def draw_graph(
    nodes, edges, title: str = "Graph", save_path=None,
):
    """Draw an undirected graph based on given nodes and edges.

    Args:
        nodes (list): The nodes of the graph.
        edges (list): The edges of the graph.
        title (str, optional): The title of the figure. Defaults to "Graph".
        save_path (str, optional): The path to save the figure. Defaults to None.
    """
    plt.figure()
    plt.title(title)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)
    nx.draw_networkx(G, pos, **OPTIONS)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path + "/{}.jpg".format(title))
    plt.show()
    return


def draw_maxcut_result(
    nodes, edges, solution_bit, title: str = "The result of MaxCut", save_path=None,
):
    """Draw an undirected graph based on given nodes and edges.

    Args:
        nodes (list): The nodes of the graph.
        edges (list): The edges of the graph.
        solution_bit (str): The result state as a binary string.
        title (str, optional): The title of the figure. Defaults to "The result of MaxCut".
        save_path (str, optional): The path to save the figure. Defaults to None.
    """
    plt.figure()
    plt.title(title)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)

    node_color = ["red" if solution_bit[v] == "1" else "#1f78b4" for v in nodes]
    edge_color = []
    edge_style = []
    for u in nodes:
        for v in range(u + 1, len(nodes)):
            if (u, v) in edges or (v, u) in edges or [u, v] in edges or [v, u] in edges:
                if solution_bit[u] == solution_bit[v]:
                    edge_color.append("black")
                    edge_style.append("solid")
                else:
                    edge_color.append("red")
                    edge_style.append("dashed")

    nx.draw(
        G,
        pos,
        node_color=node_color,
        edge_color=edge_color,
        style=edge_style,
        **OPTIONS
    )
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path + "/{}.jpg".format(title))
    plt.show()

