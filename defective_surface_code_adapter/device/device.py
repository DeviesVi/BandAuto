"""Defective surface code device describer."""

import networkx as nx
import numpy as np
import dataclasses
import pickle
from typing import List, Optional

@dataclasses.dataclass
class Device:
    """Discribe the rotated lattice where to generate stabilizer code, assuming top and bottom is x boundaries, left and right is z boundaries.
    Example: A 3*4 data qubit lattice for rotated surface code, D for data qubit, X for x syndrome, Z for z syndrome.
    y
    6|                  X
    5|      D       D       D       D
    4|  Z       X       Z       X       Z
    3|      D       D       D       D
    2|          Z       X       Z       
    1|      D       D       D       D
    0|          X               X
     ------------------------------------ x   
        0   1   2   3   4   5   6   7   8
    """

    data_height: int
    data_width: int
    qubit_defect_rate: float = 0.0
    coupler_defect_rate: float = 0.0
    exact_rate: bool = False

    def to_dict(self) -> dict:
        """Convert to dict."""
        return dataclasses.asdict(self)

    def __post_init__(self) -> None:
        # Create connectivity graph for rotated lattice.
        graph = nx.Graph()
        """graph attribute:
            node attribute: Dict
                name(str): Qubit name, begin with qubit type, example: Q1, X2, Z3.
                defective(boolean): Qubit defect flag, True for a defective qubit.
            edge attribute: Dict
                defective(boolean): Coupler defect flag, True for a defective coupler.
        """

        # Calculate the coordinate for data qubit, x measument qubit and z measurement qubit.
        data_qubit = [(x, y) for x in range(1, 2*self.data_width, 2)
                      for y in range(1, 2*self.data_height, 2)]
        x_meas_qubit = [(x, y) for y in range(0, 2*self.data_height+1, 2)
                        for x in range(2+y % 4, 2*self.data_width-1, 4)]
        z_meas_qubit = [(x, y) for y in range(2, 2*self.data_height-1, 2)
                        for x in range(0+y % 4, 2*self.data_width+1, 4)]

        all_qubit = data_qubit + x_meas_qubit + z_meas_qubit

        # Add node to the graph and set attribute for them.
        for qubits, prefix in [(data_qubit, 'D'), (x_meas_qubit, 'X'), (z_meas_qubit, 'Z')]:
            graph.add_nodes_from(qubits)
            nx.set_node_attributes(
                graph,
                {coord: {'name': f'{prefix}{index}', 'defective': False}
                    for index, coord in enumerate(qubits)}
            )

        # Add edge to the graph for all adjecent qubit, set default attribute.
        edge_set = set()
        for coord in all_qubit:
            for neighbor in [(-1, 1), (-1, -1), (1, 1), (1, -1)]:
                neighbor_coord = (
                    coord[0]+neighbor[0], coord[1]+neighbor[1])
                if neighbor_coord in all_qubit:
                    edge_set.add((coord, neighbor_coord))
        graph.add_edges_from(edge_set)
        nx.set_edge_attributes(
            graph,
            {edge: {'defective': False} for edge in edge_set}
        )

        self.graph = graph

    @staticmethod
    def _randomTrue(prob_true: float):
        return bool(np.random.choice([False, True], p=[1-prob_true, prob_true]))
    
    def add_random_defect(self, qubit_defect_rate: float, coupler_defect_rate: float, exact_rate: bool = False):
        """Add random defect to the device.
            Args:
                qubit_defect_rate: The probability of a qubit is defective.
                coupler_defect_rate: The probability of a coupler is defective.
                exact_rate: If True, the defect rate will be exactly qubit_defect_rate and coupler_defect_rate.
        """
        self.qubit_defect_rate = qubit_defect_rate
        self.coupler_defect_rate = coupler_defect_rate
        self.exact_rate = exact_rate

        if not exact_rate:
            for node in self.graph.nodes:
                self.graph.nodes[node]['defective'] = self._randomTrue(qubit_defect_rate)
            for edge in self.graph.edges:
                self.graph.edges[edge]['defective'] = self._randomTrue(coupler_defect_rate)
        else:
            qubit_defect_num = int(qubit_defect_rate * len(self.graph.nodes))
            coupler_defect_num = int(coupler_defect_rate * len(self.graph.edges))
            for node in np.random.choice(list(self.graph.nodes), qubit_defect_num, replace=False):
                self.graph.nodes[node]['defective'] = True
            for edge in np.random.choice(list(self.graph.edges), coupler_defect_num, replace=False):
                self.graph.edges[edge]['defective'] = True

    def save(self, path: str):
        """Save the device to a file.
            Args:
                path: The path to save the device.
        """
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path: str):
        """Load the device from a file.
            Args:
                path: The path to load the device.
        """
        return pickle.load(open(path, 'rb'))
    
    def __str__(self) -> str:
        return f'Device({self.data_height}, {self.data_width})'
    
    def __repr__(self) -> str:
        return f'Device({self.data_height}, {self.data_width})'