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
            # If node has been disabled, skip
            if cls._is_disabled_node(node):
                continue
            
            deformation_stack = [BoundaryNode(node=node, node_type=cls._get_boundary_node_type(node))]
            while deformation_stack:
                deformation_node = deformation_stack.pop()

                # Handle the node.
                new_nodes = cls._boundary_node_handler(deformation_node)
                # Push new nodes to stack.
                deformation_stack.extend(new_nodes)

    @classmethod
    def _boundary_node_handler(cls, node: BoundaryNode) -> List[BoundaryNode]:
        """Handler for a boundary node.
            Args:
                node: The node to handle.

            Returns:
                A list of new nodes introduced by the node."""

        # If the node type is X or Z, call syndrome handler.
        if cls._get_node_type(node) == 'X' or cls._get_node_type(node) == 'Z':
            return cls._boundary_syndrome_handler(node)
        # If the node type is D, call data qubit handler.
        if cls._get_node_type(node) == 'D':
            return cls._boundary_data_handler(node)

    @classmethod
    def _boundary_data_handler(cls, node: tuple) -> List[tuple]:
        """Handler for a boundary data.
            Args:
                node: The node to handle.

            Returns:
                A list of new nodes introduced by the node."""

        # Check if the boundary data is safe, if safe, return.
        if cls._boundary_data_safety_check(node):
            return []

        # If not safe, disable the node.
        cls._disable_node(node)

        # return undisabled neighbors.
        return cls._get_undisabled_neighbors(node)

    @classmethod
    def _boundary_syndrome_handler(cls, node: tuple) -> List[tuple]:
        """Handler for a boundary syndrome.
            Args:
                node: The node to handle.

            Returns:
                A list of new nodes introduced by the node."""

        # If this syndrome has different type with current boundary type, disable it.
        if cls.current_boundary_type != BoundaryNodeType.X and cls._get_node_type(node) == 'X' or cls.current_boundary_type != BoundaryNodeType.Z and cls._get_node_type(node) == 'Z':
            cls._disable_node(node)
            # return undisabled neighbors.
            return cls._get_undisabled_neighbors(node)

        # If this syndrome has only 1 undisabled neighbor, disable it.
        if len(cls._get_undisabled_neighbors(node)) == 1:
            cls._disable_node(node)
            # return undisabled neighbors.
            return cls._get_undisabled_neighbors(node)

    @classmethod
    def _boundary_data_safety_check(cls, node: tuple) -> bool:
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
        if cls._frontier_boundary_type(frontier) != cls.current_boundary_type:
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
                return BoundaryType.Z
            # If X is more than Z, its X type.
            elif node_types.count('X') > node_types.count('Z'):
                return BoundaryType.X
        if len(node_types) == 2:
            # If Z equal to X, its C type.
            if node_types.count('Z') == node_types.count('X'):
                return BoundaryType.C

        # If not match any type, raise exception.
        raise AdapterException(
            f'Invalid frontier type: frontier is {frontier}, node types is {node_types}.')

    # Utility functions.

    @classmethod
    def _disable_node(cls, node: tuple):
        """Disable a node.
            Args:
                node: The node to disable.
        """
        # If node already disabled, raise exception.
        if cls._is_disabled_node(node):
            raise AdapterException(f'Node {node} already disabled.')

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
