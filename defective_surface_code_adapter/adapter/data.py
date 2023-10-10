"""A data class maintain the boundary information of a device."""

import networkx as nx
import numpy as np
import dataclasses
import enum
from typing import List


class BoundaryType(enum.Enum):
    """Boundary type enum."""
    XT = 0
    XB = 1
    ZL = 2
    ZR = 3

class BoundaryNodeType(enum.Enum):
    """Boundary Node type enum."""
    X = 0
    Z = 1 
    C = 2 # Corner
    N = 3 # Not boundary

@dataclasses.dataclass
class Boundary:
    nodes: set
    boundary_type: BoundaryType

    def add_node(self, node: tuple):
        self.nodes.add(node)

@dataclasses.dataclass
class AdaptResult:
    disabled_nodes: List[tuple]
    stabilizers: List[List[tuple]]
    logical_x_data_qubits: List[tuple]
    logical_z_data_qubits: List[tuple]
    xt_boundary: List[tuple]
    xb_boundary: List[tuple]
    zl_boundary: List[tuple]
    zr_boundary: List[tuple]

    @staticmethod
    def from_dict(d: dict):
        return AdaptResult(
            disabled_nodes=d['disabled_nodes'],
            stabilizers=d['stabilizers'],
            logical_x_data_qubits=d['logical_x_data_qubits'],
            logical_z_data_qubits=d['logical_z_data_qubits'],
            xt_boundary=d['xt_boundary'],
            xb_boundary=d['xb_boundary'],
            zl_boundary=d['zl_boundary'],
            zr_boundary=d['zr_boundary']
        )