"""This file contains the traditional qubit disable rules for comparison."""

from .adapter import Adapter
from .data import BoundaryNodeType


class TraditionalAdapter(Adapter):
    """Traditional adapter class for comparison."""

    @classmethod
    def _internal_defect_handler(cls):
        """Traditional internal defect handler."""
        # Handle internal defective syndrome.
        cls._internal_defective_syndrome_handler()

        # Handle internal defective data.
        cls._internal_defective_data_handler()

        # Handle internal defective edge.
        cls._internal_defective_edge_handler()

        # Recursive cleaner for traditional adapter.
        cleaned_qubit_count = cls._recursive_cleaner()
        while cleaned_qubit_count > 0:
            cleaned_qubit_count = cls._recursive_cleaner()

    @classmethod
    def _recursive_cleaner(cls) -> int:
        """Recursive cleaner for traditional."""
        cleaned_qubit_count = 0

        cleaned_qubit_count += cls._bridge_syndrome_cleaner()
        cleaned_qubit_count += cls._weight_one_syndrome_cleaner()
        cleaned_qubit_count += cls._isolated_syndrome_cleaner()

        return cleaned_qubit_count

    @classmethod
    def _bridge_syndrome_cleaner(cls) -> int:
        """Bridge syndrome cleaner for traditional."""
        # Get all bridge syndrome nodes.
        bridge_syndromes = [
            node
            for node in cls._device.graph.nodes
            if (cls._get_node_type(node) == "X" or cls._get_node_type(node) == "Z")
            and not cls._is_disabled_node(node)
            and cls._is_bridge_syndrome(node)
        ]

        # Disable all of them.
        for node in bridge_syndromes:
            cls._internal_defective_syndrome_node_handler(node)

        return len(bridge_syndromes)

    @classmethod
    def _weight_one_syndrome_cleaner(cls) -> int:
        """Weight one syndrome cleaner for traditional."""
        # Get all weight one syndrome nodes.
        weight_one_syndromes = [
            node
            for node in cls._device.graph.nodes
            if (cls._get_node_type(node) == "X" or cls._get_node_type(node) == "Z")
            and not cls._is_disabled_node(node)
            and len(cls._get_undisabled_neighbors(node)) == 1
            and cls._get_boundary_data_type(cls._get_undisabled_neighbors(node)[0]) == BoundaryNodeType.N
            # If the syndrome node's neighbor is in boundary, do not clean it to push boundary inwards.
        ]

        # Disable all of them.
        for node in weight_one_syndromes:
            cls._internal_defective_syndrome_node_handler(node)

        return len(weight_one_syndromes)

    @classmethod
    def _isolated_syndrome_cleaner(cls) -> int:
        """Isolated syndrome cleaner for traditional."""
        # Get all undisabled syndrome node has no undisabled neighbors.
        isolated_syndromes = [
            node
            for node in cls._device.graph.nodes
            if (cls._get_node_type(node) == "X" or cls._get_node_type(node) == "Z")
            and not cls._is_disabled_node(node)
            and len(cls._get_undisabled_neighbors(node)) == 0
        ]

        # Disable all of them.
        for node in isolated_syndromes:
            cls._disable_node(node)

        return len(isolated_syndromes)

    @classmethod
    def _is_bridge_syndrome(cls, node) -> bool:
        """Check if a syndrome node is a bridge syndrome."""
        assert cls._get_node_type(node) == "X" or cls._get_node_type(node) == "Z"
        
        # Get all undisabled neighbors of the syndrome node.
        neighbors = list(cls._get_undisabled_neighbors(node))

        # Check if the syndrome node is a bridge syndrome.
        
        if len(neighbors) != 2:
            return False
        
        # Check if the two neighbors are in boundary, do not clean bridge on boundary to push boundary inwards.
        if cls._get_boundary_data_type(neighbors[0]) != BoundaryNodeType.N or cls._get_boundary_data_type(neighbors[1]) != BoundaryNodeType.N:
            return False
        
        # Check if the two neighbors share same x or y coordinate, that means they are not in diagonal line.
        if neighbors[0][0] == neighbors[1][0] or neighbors[0][1] == neighbors[1][1]:
            return False
        
        return True