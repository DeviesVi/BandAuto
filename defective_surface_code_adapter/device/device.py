"""Defective surface code device describer."""

import networkx as nx
import numpy as np
import dataclasses
import pickle
from typing import List, Optional
import hashlib

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
    
    @property
    def qubit_defect_rate(self) -> float:
        return self._qubit_defect_rate
    
    @property
    def coupler_defect_rate(self) -> float:
        return self._coupler_defect_rate
    
    @property
    def exact_rate(self) -> bool:
        return self._exact_rate

    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            'data_height': self.data_height,
            'data_width': self.data_width,
            'qubit_defect_rate': self.qubit_defect_rate,
            'coupler_defect_rate': self.coupler_defect_rate,
            'exact_rate': self.exact_rate,
            'strong_id': self.strong_id
        }

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

        self._qubit_defect_rate: float = 0.0
        self._coupler_defect_rate: float = 0.0
        self._exact_rate: bool = False

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

    def clear_all_defect(self):
        """Clear all defect on the device."""
        for node in self.graph.nodes:
            self.graph.nodes[node]['defective'] = False
        for edge in self.graph.edges:
            self.graph.edges[edge]['defective'] = False

    @staticmethod
    def _randomTrue(prob_true: float):
        return bool(np.random.choice([False, True], p=[1-prob_true, prob_true]))
    
    def add_random_defect(self, qubit_defect_rate: float, coupler_defect_rate: float, exact_rate: bool = False, clear_defect: bool = True):
        """Add random defect to the device.
            Args:
                qubit_defect_rate: The probability of a qubit is defective.
                coupler_defect_rate: The probability of a coupler is defective.
                exact_rate: If True, the defect rate will be exactly qubit_defect_rate and coupler_defect_rate.
        """
        if clear_defect:
            self.clear_all_defect()

        self._qubit_defect_rate = qubit_defect_rate
        self._coupler_defect_rate = coupler_defect_rate
        self._exact_rate = exact_rate

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

    def add_center_defect(self, diameter: int, clear_defect: bool = True):
        """Add defect on the center of the device with cirtain diameter.
            Args:
                diameter: The diameter of the defect.
        """
        assert self.data_height % 2 == 1 and self.data_width % 2 == 1, 'The device must be odd size.'
        assert diameter % 2 == 1, 'The diameter must be odd size.'
        if clear_defect:
            self.clear_all_defect()

        center = (self.data_width, self.data_height)
        radius = (diameter - 1) // 2
        coord_radius = 2 * radius
        # Add defect to odd coordinate qubit within the coord_radius.
        for x in range(center[0] - coord_radius, center[0] + coord_radius + 1, 2):
            for y in range(center[1] - coord_radius, center[1] + coord_radius + 1, 2):
                self.graph.nodes[(x, y)]['defective'] = True      

    def add_ractangle_defect(self, x: int, y: int, width: int, height: int, clear_defect: bool = True):
        """Add defect on the rectangle area.
            Args:
                x: The x coordinate of the left bottom corner of the rectangle.
                y: The y coordinate of the left bottom corner of the rectangle.
                width: The width of the rectangle.
                height: The height of the rectangle.
        """
        assert x >= 0 and y >= 0 and width >= 0 and height >= 0, 'The coordinate and size must be positive.'

        if clear_defect:
            self.clear_all_defect()

        for x in range(x, x + 2 * width, 2):
            for y in range(y, y + 2 * height, 2):
                if (x, y) in self.graph.nodes:
                    self.graph.nodes[(x, y)]['defective'] = True

    def save(self, path: str):
        """Save the device to a file.
            Args:
                path: The path to save the device.
        """
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path: str) -> 'Device':
        """Load the device from a file.
            Args:
                path: The path to load the device.
        """
        return pickle.load(open(path, 'rb'))
    
    def __str__(self) -> str:
        return f'Device({self.data_height}, {self.data_width})'
    
    def __repr__(self) -> str:
        return f'Device({self.data_height}, {self.data_width})'
    
    @property
    def strong_id(self) -> str:
        "Cryptographic hash of the device."
        return hashlib.sha256(pickle.dumps(self)).hexdigest()