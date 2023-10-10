"""Main adapter module."""
import networkx as nx
from ..device import Device
from typing import Dict, Union, List, Set, Optional
from .data import Boundary, BoundaryType, BoundaryNodeType, AdaptResult
from .exception import AdapterException


class Adapter:
    """Adapter class for device."""

    @classmethod
    def adapt_device(cls, device: Device) -> AdaptResult:
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

        cls._device = device

        cls._boundary_deformation()
        cls._interal_defect_handler()
        cls._search_stabilizers()
        cls._place_logical_operator()

        return AdaptResult.from_dict(cls.adapt_result)

    @classmethod
    def _boundary_deformation(cls):
        "Main boundary deformation function."

        # Get all type of boundary.
        cls._boundaries = {boundary_type: cls._get_boundary(
            boundary_type) for boundary_type in BoundaryType}

        # Handle all boundaries until no unsafe boundary data exist.
        while True:
            # Get all unsafe boundary data.
            unsafe_boundary_data = cls._get_unsafe_boundary_data()

            # If no unsafe boundary data, break.
            if len(unsafe_boundary_data) == 0:
                break

            # Handle all unsafe boundary data.
            for boundary_node in unsafe_boundary_data:
                # Handle the boundary data.
                cls._boundary_unsafe_data_handler(boundary_node)

    @classmethod
    def _get_unsafe_boundary_data(cls) -> List[tuple]:
        """Get all unsafe boundary data.
            Returns:
                A list of unsafe boundary data.
        """
        # Get all undisabled node in boundaries.
        boundary_nodes = set(node for boundary in cls._boundaries.values()
                          for node in boundary.nodes if not cls._is_disabled_node(node))
        return [node for node in boundary_nodes if not cls._boundary_data_safety_check(node)]

    @classmethod
    def _boundary_unsafe_data_handler(cls, node: tuple):
        """Handle boundary data.
            Args:
                node: The node to handle.

            Returns:
                A list of new boundary nodes.
        """

        cls._disable_node(node)
        # Handle neighbor syndromes in frontier.
        cls._frontier_cleaner(node)

    @classmethod
    def _frontier_cleaner(cls, node: tuple):
        """Handle frontier.
            Args:
                node: The node to handle.
        """
        node_type = cls._get_boundary_data_type(node)
        involved_boundaries = cls._node_in_boundaries(node)

        # Clean defect syndrome in frontier.
        for syndrome in cls._get_frontier(node):
            if cls._is_defective_node(syndrome):
                cls._frontier_syndrome_cleaner(syndrome, involved_boundaries)

        # If syndrome has less than 2 undisabled neighbors, clean it.
        for syndrome in cls._get_frontier(node):
            if len(cls._get_undisabled_neighbors(syndrome)) < 2:
                cls._frontier_syndrome_cleaner(syndrome, involved_boundaries)

        if node_type == BoundaryNodeType.X:
            # Clean all undisabled Z syndromes in frontier.
            for neighbor in cls._get_frontier(node):
                if cls._get_node_type(neighbor) == 'Z':
                    if not cls._is_disabled_node(neighbor):
                        cls._frontier_syndrome_cleaner(neighbor, involved_boundaries)

        elif node_type == BoundaryNodeType.Z:
            # Clean all undisabled X syndromes in frontier.
            for neighbor in cls._get_frontier(node):
                if cls._get_node_type(neighbor) == 'X':
                    if not cls._is_disabled_node(neighbor):
                        cls._frontier_syndrome_cleaner(neighbor, involved_boundaries)

        elif node_type == BoundaryNodeType.C:
            # In this situation, the node no more than 2 undisabled neighbors.
            # If has 2 undisabled neighbors, they must be X and Z syndrome, disable the one with less undisabled neighbors.
            if len(cls._get_frontier(node)) == 2:
                if len(cls._get_undisabled_neighbors(cls._get_frontier(node)[0])) < len(cls._get_undisabled_neighbors(cls._get_frontier(node)[1])):
                    cls._frontier_syndrome_cleaner(cls._get_frontier(node)[0], involved_boundaries)
                else:
                    cls._frontier_syndrome_cleaner(cls._get_frontier(node)[1], involved_boundaries)
            # If has 1 undisabled neighbors, do nothing.
        elif node_type == BoundaryNodeType.N:
            # Visiting wrong node, raise exception.
            raise AdapterException(f'Visiting wrong node {node}.')

            
    @classmethod
    def _frontier_syndrome_cleaner(cls, syndrome: tuple, involved_boundaries: List[BoundaryType]):
        """Clean frontier syndrome.
            Args:
                syndrome: The syndrome to clean.
                involved_boundaries: The boundaries to add new data.
        """
            
        # Disable the syndrome.
        cls._disable_node(syndrome)

        # Get all undisabled neighbors as new data.
        new_data = cls._get_undisabled_neighbors(syndrome)

        # Add new data to involved boundaries.
        for data in new_data:
            for boundary_type in involved_boundaries:
                cls._boundaries[boundary_type].add_node(data)


    @classmethod
    def _boundary_data_safety_check(cls, node: tuple) -> bool:
        """Check if a boundary data is safe.
            Args:
                node: The node to check.

            Returns:
                True if the boundary data is safe, False otherwise.
        """
        node_type = cls._get_boundary_data_type(node)

        # If the node is defective, its unsafe.
        if cls._is_defective_node(node):
            return False

        # If frontier is empty, its unsafe.
        if len(cls._get_frontier(node)) == 0:
            return False

        # If frontier contains defective node, its unsafe.
        if any([cls._is_defective_node(neighbor) for neighbor in cls._get_frontier(node)]):
            return False

        # If edge between this node and frontier is defective, its unsafe.
        for neighbor in cls._get_frontier(node):
            if cls._is_defective_edge(node, neighbor):
                return False

        # If frontier type not match node boundary type, its unsafe.
        if cls._frontier_type(node) != node_type:
            return False

        # If all check passed, its safe.
        return True

    @classmethod
    def _frontier_type(cls, node: tuple) -> BoundaryNodeType:
        """Compute the frontier type.
            Args:
                node: The node to compute.

            Returns:
                A frontier type.
        """
        # Get all node types in frontier.
        node_types = [cls._get_node_type(neighbor) for neighbor in cls._get_frontier(node)]
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
                return BoundaryNodeType.C

        # If does not match any type, its N type.
        return BoundaryNodeType.N

    @classmethod
    def _interal_defect_handler(cls):
        """Handle internal defects. If disabled internal node, record it."""

        # Handle internal defective syndrome.
        cls._internal_defective_syndrome_handler()

        # Handle internal defective data.
        cls._internal_defective_data_handler()

        # Handle internal defective edge.
        cls._internal_defective_edge_handler()

        # Clean isolated syndrome.
        cls._isolated_syndrome_cleaner()

    @classmethod
    def _internal_defective_syndrome_handler(cls):
        """Handle internal defective syndrome."""

        # Get all defective syndrome undisabled.
        defective_syndromes = [node for node in cls._device.graph.nodes if (cls._get_node_type(node) == 'X' or cls._get_node_type(node) == 'Z') and cls._is_defective_node(node) and not cls._is_disabled_node(node)]

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
        defective_data = [node for node in cls._device.graph.nodes if cls._get_node_type(node) == 'D' and cls._is_defective_node(node) and not cls._is_disabled_node(node)]

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
        defective_edges = [(node1, node2) for node1, node2 in cls._device.graph.edges if cls._is_defective_edge(node1, node2) and not cls._is_disabled_node(node1) and not cls._is_disabled_node(node2)]

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
        for node in [node1, node2]:
            if cls._get_node_type(node) == 'D' and not cls._is_disabled_node(node):
                cls._disable_node(node)

    @classmethod
    def _isolated_syndrome_cleaner(cls):
        """Clean isolated syndrome."""

        # Get all undisabled syndrome node has no undisabled neighbors.
        isolated_syndromes = [node for node in cls._device.graph.nodes if (cls._get_node_type(node) == 'X' or cls._get_node_type(node) == 'Z') and not cls._is_disabled_node(node) and len(cls._get_undisabled_neighbors(node)) == 0]

        # Disable all of them.
        for node in isolated_syndromes:
            cls._disable_node(node)

    @classmethod
    def _search_stabilizers(cls):
        """Search for stabilizers, including super stabilizers. """
        
        cls._visited_syndromes = set()

        # Create stabilizer search graph.
        for stabilizer_type in ['X', 'Z']:
            cls._create_stabilizer_search_graph(stabilizer_type)

            undisabled_syndromes = [node for node in cls._device.graph.nodes if cls._get_node_type(node) == stabilizer_type and not cls._is_disabled_node(node)]
            for syndrome in undisabled_syndromes:
                cls._search_stabilizers_from_syndrome(syndrome)

    @classmethod
    def _search_stabilizers_from_syndrome(cls, node: tuple):
        """Search for stabilizer from a syndrome node.
            Args:
                node: The syndrome node to search from.
        """
        # Pair syndromes using stabilizer search graph's connected components.
        # If a syndrome is already visited, skip it.
        if node in cls._visited_syndromes:
            return

        # Get all undisabled syndromes in the same connected component.
        connected_component = [node for node in nx.node_connected_component(cls._stabilizer_search_graph, node) if not cls._is_disabled_node(node)]
        cls.adapt_result['stabilizers'].append(connected_component)
        cls._visited_syndromes |= set(connected_component)
        
    @classmethod
    def _create_stabilizer_search_graph(cls, stablizer_type: Optional[str] = None):
        """Get stabilizer search graph.
            Returns:
                A stabilizer search graph.
        """
        cls._stabilizer_search_graph = nx.Graph()
        if stablizer_type is None:
            cls._stabilizer_search_graph.add_nodes_from(cls._device.graph.nodes)
        else:
            cls._stabilizer_search_graph.add_nodes_from([node for node in cls._device.graph.nodes if cls._get_node_type(node) == stablizer_type])

        # Add edges between internal disabled data nodes and its neighbors.
        # Get all internal disabled data nodes.
        internal_disabled_data_nodes = [node for node in cls._device.graph.nodes if cls._get_node_type(node) == 'D' and cls._is_disabled_node(node) and cls._get_boundary_data_type(node) == BoundaryNodeType.N]

        for node in internal_disabled_data_nodes:
            if stablizer_type is None:
                cls._stabilizer_search_graph.add_edges_from([(node, neighbor) for neighbor in cls._device.graph.neighbors(node)])
            else:
                cls._stabilizer_search_graph.add_edges_from([(node, neighbor) for neighbor in cls._device.graph.neighbors(node) if cls._get_node_type(neighbor) == stablizer_type])
        
    @classmethod
    def _place_logical_operator(cls):
        """Place logical operator. """
        for operator_type in ['X', 'Z']:
            cls._create_logical_operator_search_graph(operator_type)
            cls._search_logical_operator(operator_type)

    @classmethod
    def _search_logical_operator(cls, operator_type: str):
        """Search for logical operator. """
        if operator_type == 'X':
            # Create logical operator search graph.
            cls._create_logical_operator_search_graph(operator_type)
            # Search the shortest path from XT to XB.
            # Get undisabled XT and XB nodes.
            xt_nodes = [node for node in cls._logical_operator_search_graph.nodes if node in cls._boundaries[BoundaryType.XT].nodes and not cls._is_disabled_node(node)]
            xb_nodes = [node for node in cls._logical_operator_search_graph.nodes if node in cls._boundaries[BoundaryType.XB].nodes and not cls._is_disabled_node(node)]
            # Find the shortest path.
            shortest_path = cls._find_shortest_path(cls._logical_operator_search_graph, set(xt_nodes), set(xb_nodes))
            # If no path, raise exception.
            if shortest_path is None:
                raise AdapterException('No path from XT to XB.')
        if operator_type == 'Z':
            # Create logical operator search graph.
            cls._create_logical_operator_search_graph(operator_type)
            # Search the shortest path from ZL to ZR.
            # Get undisabled ZL and ZR nodes.
            zl_nodes = [node for node in cls._logical_operator_search_graph.nodes if node in cls._boundaries[BoundaryType.ZL].nodes and not cls._is_disabled_node(node)]
            zr_nodes = [node for node in cls._logical_operator_search_graph.nodes if node in cls._boundaries[BoundaryType.ZR].nodes and not cls._is_disabled_node(node)]
            # Find the shortest path.
            shortest_path = cls._find_shortest_path(cls._logical_operator_search_graph, set(zl_nodes), set(zr_nodes))
            # If no path, raise exception.
            if shortest_path is None:
                raise AdapterException('No path from ZL to ZR.')
            
        # Return all the data nodes in the shortest path.
        cls.adapt_result[f'logical_{operator_type.lower()}_data_qubits'] = [node for node in shortest_path if cls._get_node_type(node) == 'D']

    @classmethod
    def _create_logical_operator_search_graph(cls, operator_type: str):
        """Create logical operator search graph. """
        # A logical operator search graph is a subgraph with all undisabled data nodes and undisabled opposite type syndrome nodes.
        if operator_type == 'X':
            syndrome_type = 'Z'
        elif operator_type == 'Z':
            syndrome_type = 'X'
        
        cls._logical_operator_search_graph = nx.Graph()
        cls._logical_operator_search_graph.add_nodes_from([node for node in cls._device.graph.nodes if cls._get_node_type(node) == 'D' and not cls._is_disabled_node(node)])
        cls._logical_operator_search_graph.add_nodes_from([node for node in cls._device.graph.nodes if cls._get_node_type(node) == syndrome_type and not cls._is_disabled_node(node)])

        # Add edges to search graph if the edge in original graph.
        for node1, node2 in cls._device.graph.edges:
            if node1 in cls._logical_operator_search_graph.nodes and node2 in cls._logical_operator_search_graph.nodes:
                cls._logical_operator_search_graph.add_edge(node1, node2)

    @classmethod
    def _find_shortest_path(cls, graph: nx.Graph, source: Set[tuple], target: Set[tuple]):
        """Find the shortest path between two boundaries. """
        shortest_path = None
        for source_node in source:
            for target_node in target:
                try:
                    path = nx.shortest_path(graph, source=source_node, target=target_node)
                except(nx.NetworkXNoPath):
                    continue
                if shortest_path is None or len(path) < len(shortest_path):
                    shortest_path = path
        
        return shortest_path

    # Utility functions.

    @classmethod
    def _disable_node(cls, node: tuple):
        """Disable a node.
            Args:
                node: The node to disable.
        """
        # If node already disabled, raise exception.
        if cls._is_disabled_node(node):
            # print(f'Node {node} already disabled.')
            raise AdapterException(f'Node {node} already disabled.')
        
        cls.adapt_result['disabled_nodes'].append(node)

    @classmethod
    def _get_frontier(cls, node: tuple) -> List:
        """Get fontier of a data node. Froniter is undisabled neighbors of a data node.
            Args:
                node: The data node to get neighbors.

            Returns:
                A list of undisabled neighbors in frontier.
        """
        assert cls._get_node_type(node) == 'D', f'Node {node} is not a data node.'
        return cls._get_undisabled_neighbors(node)

    @classmethod
    def _get_undisabled_neighbors(cls, node: tuple) -> List:
        """Get undisabled neighbors of a node.
            Args:
                node: The node to get neighbors.

            Returns:
                A list of undisabled neighbors.
        """

        return [neighbor for neighbor in cls._device.graph.neighbors(node) if not cls._is_disabled_node(neighbor)]

    @classmethod
    def _is_defective_node(cls, node: tuple) -> bool:
        """Check if a node is defective.
            Args:
                node: The node to check.

            Returns:
                True if the node is defective, False otherwise.
        """

        return cls._device.graph.nodes[node]['defective']

    @classmethod
    def _is_defective_edge(cls, node1: tuple, node2: tuple) -> bool:
        """Check if an edge is defective.
            Args:
                node1: The first node of the edge.
                node2: The second node of the edge.

            Returns:
                True if the edge is defective, False otherwise.
        """

        return cls._device.graph.edges[node1, node2]['defective']

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

        return cls._device.graph.nodes[node]['name'][0]

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
    def _get_boundary_nodes(cls, boundary_type: BoundaryType) -> Set[tuple]:
        """Get boundary nodes by boundary type.
            Args:
                boundary_type: The boundary type.

            Returns:
                A set of boundary nodes.
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
    def _get_xt_boundary_nodes(cls) -> Set[tuple]:
        """Get XT boundary nodes.
            Returns:
                A list of XT boundary nodes.
        """

        return set(node for node in cls._device.graph.nodes if node[1] == 2*cls._device.data_height-1)

    @classmethod
    def _get_xb_boundary_nodes(cls) -> Set[tuple]:
        """Get XB boundary nodes.
            Returns:
                A list of XB boundary nodes.
        """

        return set(node for node in cls._device.graph.nodes if node[1] == 1)

    @classmethod
    def _get_zl_boundary_nodes(cls) -> Set[tuple]:
        """Get ZL boundary nodes.
            Returns:
                A list of ZL boundary nodes.
        """

        return set(node for node in cls._device.graph.nodes if node[0] == 1)

    @classmethod
    def _get_zr_boundary_nodes(cls) -> Set[tuple]:
        """Get ZR boundary nodes.
            Returns:
                A list of ZR boundary nodes.
        """

        return set(node for node in cls._device.graph.nodes if node[0] == 2*cls._device.data_width-1)

    @classmethod
    def _get_boundary_data_type(cls, node: tuple) -> BoundaryNodeType:
        """Get boundary node type of a node.
            Args:
                node: The node to get boundary type.

            Returns:
                A node boundary type, C for corner, the node is in both X and Z.
        """
        # If node in x boundary and z boundary, its corner.
        if node in cls._boundaries[BoundaryType.XT].nodes | cls._boundaries[BoundaryType.XB].nodes and node in cls._boundaries[BoundaryType.ZL].nodes | cls._boundaries[BoundaryType.ZR].nodes:
            return BoundaryNodeType.C
        # If node in x boundary, its x.
        elif node in cls._boundaries[BoundaryType.XT].nodes | cls._boundaries[BoundaryType.XB].nodes:
            return BoundaryNodeType.X
        # If node in z boundary, its z.
        elif node in cls._boundaries[BoundaryType.ZL].nodes | cls._boundaries[BoundaryType.ZR].nodes:
            return BoundaryNodeType.Z
        # If node not in any boundary, its not boundary.
        else:
            return BoundaryNodeType.N
        
    @classmethod
    def _node_in_boundaries(cls, node: tuple) -> List[BoundaryType]:
        """Return a list of boundary type showing which boundaries the node is in. """

        return [boundary.boundary_type for boundary in cls._boundaries.values() if node in boundary.nodes]