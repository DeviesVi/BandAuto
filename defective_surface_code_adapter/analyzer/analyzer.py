"""Analyse two main factor of defective surface code from https://arxiv.org/abs/2305.00138"""

import numpy as np
import networkx as nx
from typing import List, Tuple

from ..adapter import Adapter, TraditionalAdapter
from ..device import Device
from .data import AnalysisResult

class Analyzer:
    @classmethod
    def analyze_device(cls, device: Device, traditional_adapter: bool = False) -> AnalysisResult:
        """Analyze the device to get the two main factor of defective surface code"""

        cls._device = device
        if traditional_adapter:
            cls._adapt_result = TraditionalAdapter.adapt_device(device=device)
        else:
            cls._adapt_result = Adapter.adapt_device(device=device)

        x_shortest_distance, x_shortest_paths_count = cls._analyze_error_type('X')
        z_shortest_distance, z_shortest_paths_count = cls._analyze_error_type('Z')

        super_stabilizer_weights = [len(cls._data_in_stabilizer(stabilizer)) for stabilizer in cls._adapt_result.stabilizers if len(stabilizer) > 1]
        super_stabilizer_weights_x = [len(cls._data_in_stabilizer(stabilizer)) for stabilizer in cls._adapt_result.stabilizers if len(stabilizer) > 1 and cls._get_stabilizer_type(stabilizer) == 'X']
        super_stabilizer_weights_z = [len(cls._data_in_stabilizer(stabilizer)) for stabilizer in cls._adapt_result.stabilizers if len(stabilizer) > 1 and cls._get_stabilizer_type(stabilizer) == 'Z']

        max_stabilizer_weight, min_stabilizer_weight, me_stabilizer_weight, avg_stabilizer_weight = cls._calculate_statistics(super_stabilizer_weights)
        max_stabilizer_weight_x, min_stabilizer_weight_x, me_stabilizer_weight_x, avg_stabilizer_weight_x = cls._calculate_statistics(super_stabilizer_weights_x)
        max_stabilizer_weight_z, min_stabilizer_weight_z, me_stabilizer_weight_z, avg_stabilizer_weight_z = cls._calculate_statistics(super_stabilizer_weights_z)               

        return AnalysisResult(
            x_distance=x_shortest_distance,
            x_shortest_paths_count=x_shortest_paths_count,
            z_distance=z_shortest_distance,
            z_shortest_paths_count=z_shortest_paths_count,
            stabilizer_statistics={
                'max_stabilizer_weight': max_stabilizer_weight,
                'min_stabilizer_weight': min_stabilizer_weight,
                'me_stabilizer_weight': me_stabilizer_weight,
                'avg_stabilizer_weight': avg_stabilizer_weight,
                'max_stabilizer_weight_x': max_stabilizer_weight_x,
                'min_stabilizer_weight_x': min_stabilizer_weight_x,
                'me_stabilizer_weight_x': me_stabilizer_weight_x,
                'avg_stabilizer_weight_x': avg_stabilizer_weight_x,
                'max_stabilizer_weight_z': max_stabilizer_weight_z,
                'min_stabilizer_weight_z': min_stabilizer_weight_z,
                'me_stabilizer_weight_z': me_stabilizer_weight_z,
                'avg_stabilizer_weight_z': avg_stabilizer_weight_z,
            }
        )
    
    @classmethod
    def _calculate_statistics(cls, weights):
        if len(weights) > 0:
            return (
                np.max(weights),
                np.min(weights),
                np.median(weights),
                np.mean(weights)
            )
        else:
            return (None, None, None, None)
        
    @classmethod
    def _analyze_error_type(cls, error_type: str) -> Tuple[int, int]:
        """Analyze the device to get the shortest distance and shortest paths count of error type.
            Args:
                error_type: The error type to analyze.

            Returns:
                A tuple of shortest distance and shortest paths count.
        """

        assert error_type in ['X', 'Z']

        sources, targets = cls._get_sources_and_targets(error_type)

        shortest_distance, shortest_paths_count = cls._shortest_distance_and_path_count(sources=sources, targets=targets, error_type=error_type)

        return shortest_distance, shortest_paths_count
    
    @classmethod
    def _get_sources_and_targets(cls, error_type: str) -> Tuple[List[tuple], List[tuple]]:
        """Get sources and targets of error type.
            Args:
                error_type: The error type to get sources and targets.

            Returns:
                A tuple of sources and targets.
        """

        assert error_type in ['X', 'Z']

        # If error type is X, sources is xt, targets is xb.
        if error_type == 'X':
            sources = cls._adapt_result.xt_boundary
            targets = cls._adapt_result.xb_boundary
        # If error type is Z, sources is zl, targets is zr.
        elif error_type == 'Z':
            sources = cls._adapt_result.zl_boundary
            targets = cls._adapt_result.zr_boundary

        return sources, targets

    @classmethod
    def _create_logical_error_search_graph(cls, sources: List[tuple], targets: List[tuple], error_type: str):
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

        # Add start and end nodes.
        cls._logical_operator_search_graph.add_nodes_from(['start', 'end'])
        cls._logical_operator_search_graph.add_edges_from([('start', source) for source in sources])
        cls._logical_operator_search_graph.add_edges_from([(target, 'end') for target in targets])

    @classmethod
    def _shortest_distance_and_path_count(cls, sources: List[tuple], targets: List[tuple], error_type: str) -> Tuple[int, int]:
        """Find shortest distance and shortest paths count between sources and targets in a undirected graph with all edges weight equal to 1.
            Args:
                sources: The sources to find path.
                targets: The targets to find path.
                error_type: The error type to create logical error search graph.

            Returns:
                A tuple of shortest distance and shortest paths count.
        """

        assert error_type in ['X', 'Z']

        cls._create_logical_error_search_graph(sources=sources, targets=targets, error_type=error_type)

        # Get all shortest paths from start to end.
        shortest_paths = nx.all_shortest_paths(cls._logical_operator_search_graph, source='start', target='end')
        shortest_paths = list(shortest_paths)
        shortest_paths_count = len(shortest_paths)
        shortest_distance = len(shortest_paths[0]) - 2 # minus start and end nodes.

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
            data_qubits += cls._data_in_syndrome(syndrome)
        return data_qubits

    @classmethod
    def _data_in_syndrome(cls, syndrome: tuple) -> List[tuple]:
        """Get data qubits in syndrome.
            Args:
                syndrome: The syndrome to get data qubits.

            Returns:
                A list of undisabled data qubits neighbors to syndrome.
        """
        return [node for node in cls._device.graph.neighbors(syndrome) if cls._get_node_type(node) == 'D' and not cls._is_disabled_node(node)]

    @classmethod
    def _get_stabilizer_type(cls, stabilizer: List[tuple]) -> str:
        """Get stabilizer type.
            Args:
                stabilizer: The stabilizer to get type.

            Returns:
                A stabilizer type.
        """
        return cls._get_node_type(stabilizer[0])

    @classmethod
    def _get_node_type(cls, node: tuple) -> str:
        """Get node type.
            Args:
                node: The node to get type.

            Returns:
                A node type.
        """

        return cls._device.graph.nodes[node]['name'][0]