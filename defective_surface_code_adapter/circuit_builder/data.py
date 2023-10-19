import dataclasses
from typing import List
from functools import cached_property
from enum import Enum
class HoldingCycleOption(Enum):
    """Holding cycle options."""
    MIN = 'min'
    MAX = 'max'
    AVG = 'avg'

class U1Gate(Enum):
    """U1 gate options."""
    H = 'H' # Hadamard
    Y2 = 'Y2' # Y Pi/2 Rotation

class U2Gate(Enum):
    """U2 gate options."""
    CZ = 'CZ'
    CNOT = 'CNOT'

@dataclasses.dataclass
class BuilderOptions:
    """Options for the circuit builder."""
    syndrome_measurement_pattern = {
        'X':[[-1,1],[-1,-1],[1,1],[1,-1]],
        'Z':[[-1,1],[1,1],[-1,-1],[1,-1]],
    }
    syndrome_reset = True
    first_cycle_super_stabilizer_type = {
        '0': 'Z',
        '1': 'Z',
        '+': 'X',
        '-': 'X',
    }
    stabilizer_group_holding_cycle_option = HoldingCycleOption.MAX
    stabilizer_group_holding_cycle_ratio = 0.25
    u1gate = U1Gate.H
    u2gate = U2Gate.CZ

class Stabilizer:
    """A stabilizer."""
    stabilizers_count = 0

    def __init__(self, stablizer_type: str, syndromes: List[tuple]):
        """Constructor."""

        assert stablizer_type in ['X', 'Z'], "Stabilizer type must be one of 'X', 'Z'"

        self.id = Stabilizer.stabilizers_count
        Stabilizer.stabilizers_count += 1

        self.stabilizer_type = stablizer_type
        self.syndromes = syndromes

        self.data_qubits = []

    def __hash__(self) -> int:
        return hash(self.id)

class StabilizerGroup:
    """A stabilizer group consists of conflicting stabilizers."""
    stabilizer_groups_count = 0

    def __init__(self, stabilizers: List[Stabilizer]):
        """Constructor."""
        self.id = StabilizerGroup.stabilizer_groups_count
        StabilizerGroup.stabilizer_groups_count += 1

        self.stabilizers = stabilizers

        # These property only used for super stabilizer groups.
        self.max_holding_cycle = {
            'X': 0,
            'Z': 0,
        }
        self.current_holding_type = 'X'
        self.remaining_holding_cycle = 0

    def gen_stabilizers_for_1cycle(self):
        # Get stabilizers for this cycle with same type as current holding type.
        stabilizers = [stabilizer for stabilizer in self.stabilizers if stabilizer.stabilizer_type == self.current_holding_type]
        self.remaining_holding_cycle -= 1

        # If remaining holding cycle is 0, switch to another type.
        if self.remaining_holding_cycle == 0:
            self.current_holding_type = 'X' if self.current_holding_type == 'Z' else 'Z'
            self.remaining_holding_cycle = self.max_holding_cycle[self.current_holding_type]

        return stabilizers

    @cached_property
    def is_super_stabilizer(self) -> bool:
        """Check if it is a super stabilizer."""
        return len(self.stabilizers) > 1

    @cached_property
    def total_stabilizer_weight(self) -> int:
        """Get total stabilizer weight."""
        return sum([len(stabilizer.data_qubits) for stabilizer in self.stabilizers])
    
    @cached_property
    def total_stabilizer_weight_x(self) -> int:
        """Get total stabilizer weight of X stabilizers."""
        return sum([len(stabilizer.data_qubits) for stabilizer in self.stabilizers if stabilizer.stabilizer_type == 'X'])
    
    @cached_property
    def total_stabilizer_weight_z(self) -> int:
        """Get total stabilizer weight of Z stabilizers."""
        return sum([len(stabilizer.data_qubits) for stabilizer in self.stabilizers if stabilizer.stabilizer_type == 'Z'])
    
    @cached_property
    def max_stabilizer_weight(self) -> int:
        """Get max stabilizer weight."""
        return max([len(stabilizer.data_qubits) for stabilizer in self.stabilizers])
    
    @cached_property
    def max_stabilizer_weight_x(self) -> int:
        """Get max stabilizer weight of X stabilizers."""
        return max([len(stabilizer.data_qubits) for stabilizer in self.stabilizers if stabilizer.stabilizer_type == 'X'])
    
    @cached_property
    def max_stabilizer_weight_z(self) -> int:
        """Get max stabilizer weight of Z stabilizers."""
        return max([len(stabilizer.data_qubits) for stabilizer in self.stabilizers if stabilizer.stabilizer_type == 'Z'])
    
    @cached_property
    def min_stabilizer_weight(self) -> int:
        """Get min stabilizer weight."""
        return min([len(stabilizer.data_qubits) for stabilizer in self.stabilizers])
    
    @cached_property
    def min_stabilizer_weight_x(self) -> int:
        """Get min stabilizer weight of X stabilizers."""
        return min([len(stabilizer.data_qubits) for stabilizer in self.stabilizers if stabilizer.stabilizer_type == 'X'])
    
    @cached_property
    def min_stabilizer_weight_z(self) -> int:
        """Get min stabilizer weight of Z stabilizers."""
        return min([len(stabilizer.data_qubits) for stabilizer in self.stabilizers if stabilizer.stabilizer_type == 'Z'])
    
    @cached_property
    def avg_stabilizer_weight(self) -> int:
        """Get avg stabilizer weight."""
        return self.total_stabilizer_weight / len(self.stabilizers)
    
    @cached_property
    def avg_stabilizer_weight_x(self) -> int:
        """Get avg stabilizer weight of X stabilizers."""
        return self.total_stabilizer_weight_x / len(self.stabilizers)
    
    @cached_property
    def avg_stabilizer_weight_z(self) -> int:
        """Get avg stabilizer weight of Z stabilizers."""
        return self.total_stabilizer_weight_z / len(self.stabilizers)