"""Analyse two main factor of defective surface code from https://arxiv.org/abs/2305.00138"""

import numpy as np
import networkx as nx
from typing import List, Tuple

from ..adapter import Adapter
from ..device import Device
from .data import AnalysisResult

class Analyzer:
    @classmethod
    def analyze_device(cls, device: Device):
        """Analyze the device to get the two main factor of defective surface code"""

        cls._device = device
        cls._adapt_result = Adapter.adapt_device(device=device)

        for error_type in ['X', 'Z']:
            cls._create_logical_error_search_graph(error_type=error_type)
            if error_type == 'X':
                x_shortest_distance, x_shortest_paths_count = cls._shortest_distance_and_path_count(G=cls._logical_operator_search_graph, sources=cls._adapt_result.xt_boundary, targets=cls._adapt_result.xb_boundary)
            elif error_type == 'Z':
                z_shortest_distance, z_shortest_paths_count = cls._shortest_distance_and_path_count(G=cls._logical_operator_search_graph, sources=cls._adapt_result.zl_boundary, targets=cls._adapt_result.zr_boundary)
        return AnalysisResult(x_shortest_distance + 1, x_shortest_paths_count, z_shortest_distance + 1, z_shortest_paths_count)
    
    @classmethod
    def _create_logical_error_search_graph(cls, error_type: str):
        """Create a graph to search for logical error. """
        if error_type == 'X':
            stabilizer_type = 'Z'
        elif error_type == 'Z':
            stabilizer_type = 'X'

        cls._logical_operator_search_graph = nx.Graph()
        cls._logical_operator_search_graph.add_nodes_from([node for node in cls._device.graph.nodes if cls._get_node_type(node) == 'D' and not cls._is_disabled_node(node)])

        for stabilizer in cls._adapt_result.stabilizers:
            if cls._get_stabilizer_type(stabilizer) == stabilizer_type:
                data_qubits = cls._data_in_stabilizer(stabilizer)
                # Add edges between every data qubits.
                cls._logical_operator_search_graph.add_edges_from([(data_qubits[i], data_qubits[j]) for i in range(len(data_qubits)) for j in range(i+1, len(data_qubits))])          


    @classmethod
    def _shortest_distance_and_path_count(cls, G: nx.Graph, sources: List[tuple], targets: List[tuple]) -> Tuple[int, int]:
        """Find shortest distance and shortest paths count between sources and targets in a undirected graph with all edges weight equal to 1.
            Args:
                sources: The sources to find path.
                targets: The targets to find path.

            Returns:
                A tuple of shortest distance and shortest paths count.
        """
        shortest_paths = []
        shortest_distance = np.inf
        for source in sources:
            for target in targets:
                # Get shortest distance between source and target.
                distance = nx.shortest_path_length(G=G, source=source, target=target)
                if distance < shortest_distance:
                    shortest_distance = distance
                    shortest_paths = []
                if distance == shortest_distance:
                    shortest_paths += nx.all_shortest_paths(G=G, source=source, target=target)
        shortest_paths_count = len(shortest_paths)
        return shortest_distance, shortest_paths_count

    # Utility functions
    @classmethod
    def _is_disabled_node(cls, node: tuple) -> bool:
        """Check if the node is disabled"""
        return node in cls._adapt_result.disabled_nodes

    @classmethod
    def _data_in_stabilizer(cls, stabilizer: List[tuple]) -> List[tuple]:
        """Get data qubits in stabilizer.
            Args:
                stabilizer: The stabilizer to get data qubits.

            Returns:
                A list of undisabled data qubits neighbors to syndromes in stabilizer.
        """
        data_qubits = []
        for syndrome in stabilizer:
            data_qubits += [node for node in cls._device.graph.neighbors(syndrome) if cls._get_node_type(node) == 'D' and not cls._is_disabled_node(node)]
        return data_qubits

    @classmethod
    def _get_stabilizer_type(cls, stabilizer: List[tuple]) -> str:
        """Get stabilizer type.
            Args:
                stabilizer: The stabilizer to get type.

            Returns:
                A stabilizer type.
        """
        return cls._device.graph.nodes[stabilizer[0]]['name'][0]

    @classmethod
    def _get_node_type(cls, node: tuple) -> str:
        """Get node type.
            Args:
                node: The node to get type.

            Returns:
                A node type.
        """

        return cls._device.graph.nodes[node]['name'][0]