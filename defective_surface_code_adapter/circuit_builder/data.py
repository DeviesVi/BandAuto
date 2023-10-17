import dataclasses
from typing import List

class BuilderOptions:
    """Options for the circuit builder."""
    first_cycle_super_stabilizer_type = {
        '0': 'Z',
        '1': 'Z',
        '+': 'X',
        '-': 'X',
    }


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

class StabilizerGroup:
    """A stabilizer group consists of conflicting stabilizers."""
    stabilizer_groups_count = 0

    def __init__(self, stabilizers: List[Stabilizer]):
        """Constructor."""
        self.id = StabilizerGroup.stabilizer_groups_count
        StabilizerGroup.stabilizer_groups_count += 1

        self.stabilizers = stabilizers