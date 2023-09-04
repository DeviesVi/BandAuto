"""Main adapter module."""
import networkx as nx
from ..device import Device
from typing import Dict, Union, List, Set
from .data import Boundary, BoundaryType, BoundaryNode, BoundaryNodeType
from .exception import AdapterException


class Adapter:
    """Adapter class for device."""

    @classmethod
    def adapt_device(cls, device: Device) -> Dict[str, List]:
        """Adapt a device.
            Args:
                device: The device to adapt.

            Returns:
                A dict with keys:
                    disabled_nodes: A list of disabled nodes.
                    stabilizers: A list of stabilizers.
                    logical_x_data_qubits: A list of logical x data qubits.
                    logical_z_data_qubits: A list of logical z data qubits.
        """

        cls.adapt_result = {
            'disabled_nodes': [],
            'stabilizers': [],
            'logical_x_data_qubits': [],
            'logical_z_data_qubits': [],
        }

        cls.device = device

        cls._boundary_deformation()
        cls._interal_defect_handler()
        cls._search_stabilizers()

        return cls.adapt_result

    @classmethod
    def _boundary_deformation(cls):
        "Main boundary deformation function."

        # Get all type of boundary.
        cls.boundaries = {boundary_type: cls._get_boundary(
            boundary_type) for boundary_type in BoundaryType}

        # Get all nodes in boundaries.
        boundary_nodes = set()
        for boundary in cls.boundaries.values():
            boundary_nodes.update(boundary.nodes)

        # Handle all boundary nodes.
        for node in boundary_nodes:
            deformation_stack: List[BoundaryNode] = [BoundaryNode(
                node=node, node_type=cls._get_boundary_node_type(node))]
            while deformation_stack:
                current_boundary_node = deformation_stack.pop()

                # Handle the node.
                new_boundary_nodes = cls._boundary_node_handler(
                    current_boundary_node)
                # Push new nodes to stack.
                deformation_stack.extend(new_boundary_nodes)

    @classmethod
    def _boundary_node_handler(cls, boundary_node: BoundaryNode) -> List[BoundaryNode]:
        """Handler for a boundary node.
            Args:
                boundary_node: The boundary node to handle.

            Returns:
                A list of new boundary nodes introduced by the boundary node.
        """

        node = boundary_node.node
        # If node has been disabled, return empty
        if cls._is_disabled_node(node):
            return []
        # If the node type is X or Z, call syndrome handler.
        if cls._get_node_type(node) == 'X' or cls._get_node_type(node) == 'Z':
            return cls._boundary_syndrome_handler(boundary_node)
        # If the node type is D, call data qubit handler.
        if cls._get_node_type(node) == 'D':
            return cls._boundary_data_handler(boundary_node)

    @classmethod
    def _boundary_data_handler(cls, boundary_node: BoundaryNode) -> List[BoundaryNode]:
        """Handler for a boundary data.
            Args:
                boundary_node: The boundary node to handle.

            Returns:
                A list of new boundary nodes introduced by the boundary node.
        """
        node = boundary_node.node

        # Check if the boundary data is safe, if safe, return.
        if cls._boundary_data_safety_check(boundary_node):
            return []

        # If not safe, disable the node.
        cls._disable_node(node)
        # Introduce new boundary nodes.
        return cls._introduce_new_boundary_nodes(boundary_node)

    @classmethod
    def _boundary_syndrome_handler(cls, boundary_node: BoundaryNode) -> List[BoundaryNode]:
        """Handler for a boundary syndrome.
            Args:
                boundary_node: The boundary node to handle.

            Returns:
                A list of new boundary nodes introduced by the boundary node.
        """
        node = boundary_node.node

        # Check if the boundary syndrome is safe, if safe, return.
        if cls._boundary_syndrome_safety_check(boundary_node):
            return []
        
        # If not safe, disable the node.
        cls._disable_node(node)
        # Introduce new boundary nodes.
        return cls._introduce_new_boundary_nodes(boundary_node)

    @classmethod
    def _introduce_new_boundary_nodes(cls, boundary_node: BoundaryNode) -> List[BoundaryNode]:
        new_boundary_nodes = []
        node = boundary_node.node
        # Get all undisabled neighbors.
        frontier = cls._get_undisabled_neighbors(node)
        for neighbor in frontier:
            if cls._get_boundary_node_type(neighbor) == BoundaryNodeType.N:
                new_boundary_nodes.append(BoundaryNode(
                    node=neighbor, node_type=boundary_node.node_type))
            else:
                new_boundary_nodes.append(BoundaryNode(
                    node=neighbor, node_type=cls._get_boundary_node_type(neighbor)))
        return new_boundary_nodes

    @classmethod
    def _boundary_data_safety_check(cls, boudnary_node: BoundaryNode) -> bool:
        node = boudnary_node.node

        # If the node is defective, its unsafe.
        if cls._is_defective_node(node):
            return False

        # Get all undisabled neighbors.
        frontier = cls._get_undisabled_neighbors(node)

        # If frontier contains defective node, its unsafe.
        if any([cls._is_defective_node(neighbor) for neighbor in frontier]):
            return False

        # If edge between this node and frontier is defective, its unsafe.
        for neighbor in frontier:
            if cls._is_defective_edge(node, neighbor):
                return False

        # If frontier type not match current boundary type, its unsafe.
        if cls._frontier_boundary_type(frontier) != boudnary_node.node_type:
            # If none of frontier type or boundary node type is C, its unsafe
            if cls._frontier_boundary_type(frontier) != BoundaryNodeType.C and boudnary_node.node_type != BoundaryNodeType.C:
                return False

        # If all check passed, its safe.
        return True

    @classmethod
    def _boundary_syndrome_safety_check(cls, boundary_node: BoundaryNode) -> bool:
        node = boundary_node.node
        # If this syndrome has different type with boundary node type, its unsafe.
        if boundary_node.node_type != BoundaryNodeType.X and cls._get_node_type(node) == 'X' or boundary_node.node_type != BoundaryNodeType.Z and cls._get_node_type(node) == 'Z':
            return False

        # If this syndrome has less than 2 undisabled neighbor, its unsafe.
        if len(cls._get_undisabled_neighbors(node)) < 2:
            return False
        
        # If all check passed, its safe.
        return True

    @classmethod
    def _frontier_boundary_type(cls, frontier: List[tuple]):
        """Compute the frontier type.
            Args:
                frontier: The frontier to compute.

            Returns:
                A frontier type.
        """
        # Get all node types in frontier.
        node_types = [cls._get_node_type(node) for node in frontier]
        if len(node_types) == 3:
            # If Z is more than X, its Z type.
            if node_types.count('Z') > node_types.count('X'):
                return BoundaryNodeType.Z
            # If X is more than Z, its X type.
            elif node_types.count('X') > node_types.count('Z'):
                return BoundaryNodeType.X
        if len(node_types) == 2:
            # If Z equal to X, its C type.
            if node_types.count('Z') == node_types.count('X'):
                # If the two neighbors one has 2 undisabled neighbors, its C type.
                if len(cls._get_undisabled_neighbors(frontier[0])) == 2 or len(cls._get_undisabled_neighbors(frontier[1])) == 2:
                    return BoundaryNodeType.C

        # If does not match any type, its N type.
        return BoundaryNodeType.N

    # Utility functions.

    @classmethod
    def _disable_node(cls, node: tuple):
        """Disable a node.
            Args:
                node: The node to disable.
        """
        # If node already disabled, raise exception.
        if cls._is_disabled_node(node):
            print(f'Node {node} already disabled.')
            # raise AdapterException(f'Node {node} already disabled.')
        
        cls.adapt_result['disabled_nodes'].append(node)

    @classmethod
    def _get_undisabled_neighbors(cls, node: tuple) -> List:
        """Get undisabled neighbors of a node.
            Args:
                node: The node to get neighbors.

            Returns:
                A list of undisabled neighbors.
        """

        return [neighbor for neighbor in cls.device.graph.neighbors(node) if not cls._is_disabled_node(neighbor)]

    @classmethod
    def _is_defective_node(cls, node: tuple) -> bool:
        """Check if a node is defective.
            Args:
                node: The node to check.

            Returns:
                True if the node is defective, False otherwise.
        """

        return cls.device.graph.nodes[node]['defective']

    @classmethod
    def _is_defective_edge(cls, node1: tuple, node2: tuple) -> bool:
        """Check if an edge is defective.
            Args:
                node1: The first node of the edge.
                node2: The second node of the edge.

            Returns:
                True if the edge is defective, False otherwise.
        """

        return cls.device.graph.edges[node1, node2]['defective']

    @classmethod
    def _is_disabled_node(cls, node: tuple) -> bool:
        """Check if a node is disabled.
            Args:
                node: The node to check.

            Returns:
                True if the node is disabled, False otherwise.
        """

        return node in cls.adapt_result['disabled_nodes']

    @classmethod
    def _get_node_type(cls, node):
        """Get node type.
            Args:
                node: The node to get type.

            Returns:
                A node type.
        """

        return cls.device.graph.nodes[node]['name'][0]

    @classmethod
    def _get_boundary(cls, boundary_type: BoundaryType) -> Boundary:
        """Get boundary by boundary type.
            Args:
                boundary_type: The boundary type.

            Returns:
                A boundary.
        """

        return Boundary(
            nodes=cls._get_boundary_nodes(boundary_type),
            boundary_type=boundary_type,
        )

    @classmethod
    def _get_boundary_nodes(cls, boundary_type: BoundaryType) -> List:
        """Get boundary nodes by boundary type.
            Args:
                boundary_type: The boundary type.

            Returns:
                A list of boundary nodes.
        """

        if boundary_type == BoundaryType.XT:
            return cls._get_xt_boundary_nodes()
        elif boundary_type == BoundaryType.XB:
            return cls._get_xb_boundary_nodes()
        elif boundary_type == BoundaryType.ZL:
            return cls._get_zl_boundary_nodes()
        elif boundary_type == BoundaryType.ZR:
            return cls._get_zr_boundary_nodes()
        else:
            raise ValueError(f'Invalid boundary type: {boundary_type}.')

    @classmethod
    def _get_xt_boundary_nodes(cls) -> List:
        """Get XT boundary nodes.
            Returns:
                A list of XT boundary nodes.
        """

        return [node for node in cls.device.graph.nodes if node[1] == 2*cls.device.data_height-1]

    @classmethod
    def _get_xb_boundary_nodes(cls) -> List:
        """Get XB boundary nodes.
            Returns:
                A list of XB boundary nodes.
        """

        return [node for node in cls.device.graph.nodes if node[1] == 1]

    @classmethod
    def _get_zl_boundary_nodes(cls) -> List:
        """Get ZL boundary nodes.
            Returns:
                A list of ZL boundary nodes.
        """

        return [node for node in cls.device.graph.nodes if node[0] == 1]

    @classmethod
    def _get_zr_boundary_nodes(cls) -> List:
        """Get ZR boundary nodes.
            Returns:
                A list of ZR boundary nodes.
        """

        return [node for node in cls.device.graph.nodes if node[0] == 2*cls.device.data_width-1]

    @classmethod
    def _get_boundary_node_type(cls, node: tuple) -> BoundaryNodeType:
        """Get boundary node type of a node.
            Args:
                node: The node to get boundary type.

            Returns:
                A node boundary type, C for corner, the node is in both X and Z.
        """
        # If node in x boundary and z boundary, its corner.
        if node in cls.boundaries[BoundaryType.XT].nodes + cls.boundaries[BoundaryType.XB].nodes and node in cls.boundaries[BoundaryType.ZL].nodes + cls.boundaries[BoundaryType.ZR].nodes:
            return BoundaryNodeType.C
        # If node in x boundary, its x.
        elif node in cls.boundaries[BoundaryType.XT].nodes + cls.boundaries[BoundaryType.XB].nodes:
            return BoundaryNodeType.X
        # If node in z boundary, its z.
        elif node in cls.boundaries[BoundaryType.ZL].nodes + cls.boundaries[BoundaryType.ZR].nodes:
            return BoundaryNodeType.Z
        # If node not in any boundary, its not boundary.
        else:
            return BoundaryNodeType.N

    @classmethod
    def _interal_defect_handler(cls):
        """Handle internal defects."""

        # Handle internal defective syndrome.
        cls._internal_defective_syndrome_handler()

        # Handle internal defective data.
        cls._internal_defective_data_handler()

        # Handle internal defective edge.
        cls._internal_defective_edge_handler()

    @classmethod
    def _internal_defective_syndrome_handler(cls):
        """Handle internal defective syndrome."""

        # Get all defective syndrome undisabled.
        defective_syndromes = [node for node in cls.device.graph.nodes if (cls._get_node_type(node) == 'X' or cls._get_node_type(node) == 'Z') and cls._is_defective_node(node) and not cls._is_disabled_node(node)]

        # Handle all defective syndrome.
        for node in defective_syndromes:
            cls._internal_defective_syndrome_node_handler(node)

    @classmethod
    def _internal_defective_syndrome_node_handler(cls, node: tuple):
        """Handle internal defective syndrome node.
            Args:
                node: The node to handle.
        """

        # Disabled syndrome and its neighbors. According to https://arxiv.org/ftp/arxiv/papers/1208/1208.0928.pdf
        cls._disable_node(node)
        for neighbor in cls._get_undisabled_neighbors(node):
            cls._disable_node(neighbor)

    @classmethod
    def _internal_defective_data_handler(cls):
        """Handle internal defective data."""

        # Get all defective data undisabled.
        defective_data = [node for node in cls.device.graph.nodes if cls._get_node_type(node) == 'D' and cls._is_defective_node(node) and not cls._is_disabled_node(node)]

        # Handle all defective data.
        for node in defective_data:
            cls._internal_defective_data_node_handler(node)
        
    @classmethod
    def _internal_defective_data_node_handler(cls, node: tuple):
        """Handle internal defective data node.
            Args:
                node: The node to handle.
        """

        # Disable data only.
        cls._disable_node(node)

    @classmethod
    def _internal_defective_edge_handler(cls):
        """Handle internal defective edge."""

        # Get all defective edge with 2 undisabled nodes.
        defective_edges = [(node1, node2) for node1, node2 in cls.device.graph.edges if cls._is_defective_edge(node1, node2) and not cls._is_disabled_node(node1) and not cls._is_disabled_node(node2)]

        # Handle all defective edge.
        for node1, node2 in defective_edges:
            cls._internal_defective_edge_node_handler(node1, node2)

    @classmethod
    def _internal_defective_edge_node_handler(cls, node1: tuple, node2: tuple):
        """Handle internal defective edge node.
            Args:
                node1: The first node of the edge.
                node2: The second node of the edge.
        """

        # Disable the data node.
        if cls._get_node_type(node1) == 'D':
            cls._disable_node(node1)
        elif cls._get_node_type(node2) == 'D':
            cls._disable_node(node2)

    @classmethod
    def _search_stabilizers(cls):
        """Search for stabilizers, including super stabilizers. """

        pass