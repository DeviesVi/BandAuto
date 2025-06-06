from collections import defaultdict
import dataclasses
from math import floor
from typing import List, Dict, Callable
from functools import cached_property
from enum import Enum, auto

class HoldingCycleOption(Enum):
    """Holding cycle options."""
    class Value(Enum):
        MIN = auto()
        MAX = auto()
        AVG = auto()
        ME = auto()
        SPECIFIED = auto() # Specified by stabilizer_group_holding_cycle_ratio
    
    class Scope(Enum):
        GLOBAL = auto()
        LOCAL = auto()

    class XZ_Separate(Enum):
        SEPARATE = auto()
        SAME = auto()

    GLOBALAVG = (Value.AVG, Scope.GLOBAL, XZ_Separate.SAME)
    GLOBALAVG_SEPARATE = (Value.AVG, Scope.GLOBAL, XZ_Separate.SEPARATE)

    LOCALAVG = (Value.AVG, Scope.LOCAL, XZ_Separate.SAME)
    LOCALAVG_SEPARATE = (Value.AVG, Scope.LOCAL, XZ_Separate.SEPARATE)
    
    LOCALMAX = (Value.MAX, Scope.LOCAL, XZ_Separate.SAME)
    LOCALMAX_SEPARATE = (Value.MAX, Scope.LOCAL, XZ_Separate.SEPARATE)
    
    SPECIFIED = (Value.SPECIFIED, Scope.GLOBAL, XZ_Separate.SEPARATE)


class OPType(Enum):
    INIT = 'INIT'
    U1 = 'U1'
    U2 = 'U2'
    MEAS = 'MEAS'

class U1Gate(Enum):
    """U1 gate options."""
    H = 'H' # Hadamard
    Y2 = 'Y2' # Y Pi/2 Rotation

class U2Gate(Enum):
    """U2 gate options."""
    CZ = 'CZ'
    CNOT = 'CNOT'

@dataclasses.dataclass
class PhysicalErrors:
    u1: float
    u2: float
    idle: float
    readout_idle: float
    reset: float
    measurement: float
    
    @staticmethod
    def SI1000_from_p(p: float) -> 'PhysicalErrors':
        """Get SI1000 error model from p.
            SI1000 from: https://arxiv.org/abs/2202.11845
        """
        return PhysicalErrors(
            u1 = 0.1 * p,
            u2 = p,
            idle = 0.1 * p,
            readout_idle = 2 * p,
            reset = 2 * p,
            measurement = 5 * p,
        )
    
    @staticmethod
    def ratio_google_error(ratio: float) -> 'PhysicalErrors':
        """Get error model from Google's Nature.
            https://www.nature.com/articles/s41586-022-05434-1
            experimental operation point
        """
        return PhysicalErrors(
            u1 = 1.09e-3 * ratio,
            u2 = 6.05e-3 * ratio,
            idle = 0,
            readout_idle = 2.46e-2 * ratio,
            reset = 1.86e-3 * ratio,
            measurement = 1.96e-2 * ratio,
        )
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dict."""
        return dataclasses.asdict(self)

@dataclasses.dataclass
class BuilderOptions:
    """Options for the circuit builder."""

    syndrome_measurement_pattern = {
        'X':[[-1,1],[-1,-1],[1,1],[1,-1]],
        'Z':[[-1,1],[1,1],[-1,-1],[1,-1]],
    }

    syndrome_reset = True
    
    first_cycle_super_stabilizer_type = {
        '0': 'X',
        '1': 'X',
        '+': 'Z',
        '-': 'Z',
    } # Other types are unimplemented; do not change this option. This setup is for getting first round infomation of the undetermined stabilizers.

    data_measurment_stabilizer_type = {
        '0': 'Z',
        '1': 'Z',
        '+': 'X',
        '-': 'X',
    } # Do not modify this option.

    stabilizer_group_holding_cycle_option = HoldingCycleOption.SPECIFIED
    stabilizer_group_holding_cycle_ratio = 0.35
    stabilizer_group_holding_cycle_offset = 0.000
    stabilizer_group_specified_holding_cycle = 1

    u1gate = U1Gate.H
    u2gate = U2Gate.CZ

    use_traditional_adapter = False

    physical_errors = PhysicalErrors.ratio_google_error(1)

    def __post_init__(self) -> None:
        """Post init."""
        self._stabilizer_group_holding_cycle_ratio_x: float | None = None
        self._stabilizer_group_holding_cycle_ratio_z: float | None = None
        self._stabilizer_group_specified_holding_cycle_x: int | None = None
        self._stabilizer_group_specified_holding_cycle_z: int | None = None

    @property
    def stabilizer_group_holding_cycle_ratio_x(self) -> float:
        if self._stabilizer_group_holding_cycle_ratio_x is None:
            return self.stabilizer_group_holding_cycle_ratio
        
        return self._stabilizer_group_holding_cycle_ratio_x
    
    @stabilizer_group_holding_cycle_ratio_x.setter
    def stabilizer_group_holding_cycle_ratio_x(self, value: float):
        self._stabilizer_group_holding_cycle_ratio_x = value

    @property
    def stabilizer_group_holding_cycle_ratio_z(self) -> float:
        if self._stabilizer_group_holding_cycle_ratio_z is None:
            return self.stabilizer_group_holding_cycle_ratio
        
        return self._stabilizer_group_holding_cycle_ratio_z
    
    @stabilizer_group_holding_cycle_ratio_z.setter
    def stabilizer_group_holding_cycle_ratio_z(self, value: float):
        self._stabilizer_group_holding_cycle_ratio_z = value

    def stabilizer_group_holding_cycle_function(self, weight: int, ratio: float, offset: float) -> int:
        """Get stabilizer group holding cycle from linear function."""
        return floor(weight * ratio + offset)
    
    @property
    def stabilizer_group_specified_holding_cycle_x(self) -> int:
        if self._stabilizer_group_specified_holding_cycle_x is None:
            return self.stabilizer_group_specified_holding_cycle
        
        return self._stabilizer_group_specified_holding_cycle_x
    
    @stabilizer_group_specified_holding_cycle_x.setter
    def stabilizer_group_specified_holding_cycle_x(self, value: int):
        self._stabilizer_group_specified_holding_cycle_x = value

    @property
    def stabilizer_group_specified_holding_cycle_z(self) -> int:
        if self._stabilizer_group_specified_holding_cycle_z is None:
            return self.stabilizer_group_specified_holding_cycle
        
        return self._stabilizer_group_specified_holding_cycle_z
    
    @stabilizer_group_specified_holding_cycle_z.setter
    def stabilizer_group_specified_holding_cycle_z(self, value: int):
        self._stabilizer_group_specified_holding_cycle_z = value
    

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

    @cached_property
    def is_super_stabilizer(self) -> bool:
        """Check if it is a super stabilizer."""
        return len(self.syndromes) > 1

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
            'X': None,
            'Z': None,
        }
        self.current_holding_type = None
        self.current_holding_cycle = None

    def init(self, max_holding_cycle_x: float, max_holding_cycle_z: float, first_cycle_super_stabilizer_type: str):
        """Initialize the stabilizer group."""
        self.max_holding_cycle['X'] = max_holding_cycle_x
        self.max_holding_cycle['Z'] = max_holding_cycle_z
        self.current_holding_type = first_cycle_super_stabilizer_type
        self.current_holding_cycle = 0

    def gen_stabilizers_for_1cycle(self):
        if self.is_super_stabilizer_group:
            # Get stabilizers for this cycle with same type as current holding type.
            self._this_cycle_stabilizers = [stabilizer for stabilizer in self.stabilizers if stabilizer.stabilizer_type == self.current_holding_type]
            self.current_holding_cycle += 1

            # If current holding cycle exceeds max holding cycle, switch to the other type.
            assert isinstance(self.max_holding_cycle[self.current_holding_type], int), f"Max holding cycle for stabilizer type {self.current_holding_type} is not an integer."
            if self.current_holding_cycle >= self.max_holding_cycle[self.current_holding_type]:
                self.current_holding_type = 'Z' if self.current_holding_type == 'X' else 'X'
                self.current_holding_cycle = 0
        else:
            self._this_cycle_stabilizers = self.stabilizers
        return self._this_cycle_stabilizers
    
    @property
    def this_cycle_stabilizers(self) -> List[Stabilizer]:
        """Get stabilizers for this cycle."""
        return self._this_cycle_stabilizers

    @cached_property
    def is_super_stabilizer_group(self) -> bool:
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
    def total_stabilizer_count(self) -> int:
        """Get total stabilizer count."""
        return len(self.stabilizers)
    
    @cached_property
    def total_stabilizer_count_x(self) -> int:
        """Get total stabilizer count of X stabilizers."""
        return len([stabilizer for stabilizer in self.stabilizers if stabilizer.stabilizer_type == 'X'])
    
    @cached_property
    def total_stabilizer_count_z(self) -> int:
        """Get total stabilizer count of Z stabilizers."""
        return len([stabilizer for stabilizer in self.stabilizers if stabilizer.stabilizer_type == 'Z'])

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
    def avg_stabilizer_weight(self) -> float:
        """Get avg stabilizer weight."""
        return self.total_stabilizer_weight / self.total_stabilizer_count
    
    @cached_property
    def avg_stabilizer_weight_x(self) -> float:
        """Get avg stabilizer weight of X stabilizers."""
        return self.total_stabilizer_weight_x / self.total_stabilizer_count_x
    
    @cached_property
    def avg_stabilizer_weight_z(self) -> float:
        """Get avg stabilizer weight of Z stabilizers."""
        return self.total_stabilizer_weight_z / self.total_stabilizer_count_z
    

@dataclasses.dataclass
class NodeRecord:
    """A syndrome record."""
    node: tuple
    cycle: int

    def __hash__(self) -> int:
        return hash((self.node, self.cycle))
    
@dataclasses.dataclass
class MeasurementRecords:
    measurment_count: int = 0
    node_measurment_index: Dict[NodeRecord, int] = dataclasses.field(default_factory=dict)

    def add_record(self, node: tuple, cycle: int):
        """Add a measurement record."""
        self.node_measurment_index[NodeRecord(node, cycle)] = self.measurment_count
        self.measurment_count += 1

    def stim_rec_index(self, node: tuple, cycle: int) -> int:
        """Get stim record index."""
        return self.node_measurment_index[NodeRecord(node, cycle)] - self.measurment_count
    
