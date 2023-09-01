import networkx as nx
import matplotlib.pyplot as plt
from typing import List


def plot_graph(graph: nx.Graph, disabled_nodes: List[tuple]):

    # Set node color. Black for disabled node, red for defective node, yellow for x syndrome node, green for z syndrome node, blue for data node.
    node_color = [
        '#000000' if node in disabled_nodes
        else '#fa0000' if graph.nodes[node]['defective'] == True
        else '#ffb803' if graph.nodes[node]['name'][0] == 'X'
        else '#03ff42' if graph.nodes[node]['name'][0] == 'Z'
        else '#1f78b4' for node in graph.nodes
    ]
    # Set edge color. Red for defective edge, blue for normal edge.
    edge_color = [
        '#fa0000' if graph[node1][node2]['defective'] == True else '#1f78b4'
        for node1, node2 in graph.edges
    ]

    # draw graph
    nx.draw_networkx(
        graph,
        {n: n for n in nx.nodes(graph)},
        with_labels=False,
        node_size=100,
        node_color=node_color,
        edge_color=edge_color
    )