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
class BoundaryNode:
    node: tuple
    node_type: BoundaryNodeType

@dataclasses.dataclass
class Boundary:
    nodes: List
    boundary_type: BoundaryType