"""Base constructor to build circuits for the surface code."""

from abc import ABC, abstractmethod
from typing import List

from ..device import Device
from ..adapter import Adapter

from data import BuilderOptions, Stabilizer


class BaseBuilder(ABC):
    """Base constructor to build circuits for the surface code."""

    def __init__(self, device:Device) -> None:
        """Constructor."""
        self.device = device
        self.adapt_result = Adapter.adapt_device(device)
        self._stabilizers = []

        # Builder preprocessing
        self._prepare_stabilizers(self.adapt_result.stabilizers)
        self._prepare_stabilizer_groups()

    def _prepare_stabilizers(self, stabilizers):      
        for stabilizer in stabilizers:
            # Prepare stabilizer
            self._stabilizers.append(self._prepare_stabilizer(stabilizer))

    def _prepare_stabilizer(self, stabilizer) -> Stabilizer:
        """Prepare a stabilizer."""
        stabilizer_ = Stabilizer(self._get_stabilizer_type(stabilizer), stabilizer)
        stabilizer_.data_qubits = self._data_in_stabilizer(stabilizer)
        return Stabilizer
    
    def _prepare_stabilizer_groups(self):
        """Prepare stabilizer groups."""
        



    def build(self, ec_cycle: int, initial_state: str):
        """Build the circuit."""

        assert initial_state in ['0', '1', '+', '-'], "Initial state must be one of '0', '1', '+', '-'"

        self.init_circuit()
        self.state_preparation(initial_state)

    # Utility methods
    def _is_disabled_node(self, node: tuple) -> bool:
        """Check if the node is disabled"""
        return node in self.adapt_result.disabled_nodes

    def _data_in_stabilizer(self, stabilizer: List[tuple]) -> List[tuple]:
        """Get data qubits in stabilizer.
            Args:
                stabilizer: The stabilizer to get data qubits.

            Returns:
                A list of undisabled data qubits neighbors to syndromes in stabilizer.
        """
        data_qubits = []
        for syndrome in stabilizer:
            data_qubits += self._data_in_syndrome(syndrome)
        return data_qubits
    
    def _data_in_syndrome(self, syndrome: tuple) -> List[tuple]:
        """Get data qubits in syndrome.
            Args:
                syndrome: The syndrome to get data qubits.

            Returns:
                A list of undisabled data qubits neighbors to syndrome.
        """
        return [node for node in self.device.graph.neighbors(syndrome) if self._get_node_type(node) == 'D' and not self._is_disabled_node(node)]

    def _get_stabilizer_type(self, stabilizer: List[tuple]) -> str:
        """Get stabilizer type.
            Args:
                stabilizer: The stabilizer to get type.

            Returns:
                A stabilizer type.
        """
        return self._get_node_type(stabilizer[0])

    def _get_node_type(self, node: tuple) -> str:
        """Get node type.
            Args:
                node: The node to get type.

            Returns:
                A node type.
        """

        return self.device.graph.nodes[node]['name'][0]
    
    def _is_syndromes_conflict(self, syndromes1: tuple, syndromes2: tuple) -> bool:
        """Check if two syndromes conflict.
            Args:
                syndromes1: The first syndrome.
                syndromes2: The second syndrome.

            Returns:
                True if two syndromes conflict, False otherwise.
        """
        # If two syndromes are of the same type, they are not conflict.
        if self._get_node_type(syndromes1) == self._get_node_type(syndromes2):
            return False
        # If two syndromes share odd number of data qubits, they are conflict.
        return len(set(self._data_in_syndrome(syndromes1)) & set(self._data_in_syndrome(syndromes2))) % 2 == 1

    # Abstract methods
    @abstractmethod
    def init_circuit(self):
        """Initialize the circuit."""
        pass

    @abstractmethod
    def state_preparation(self, initial_state):
        """Prepare the state."""
        pass

    @abstractmethod
    def unitary1(self, dest):
        """Insert single qubit unitary."""
        pass

    @abstractmethod
    def unitary2(self, dest1, dest2):
        """Insert two qubit unitary."""
        pass

    @abstractmethod
    def measurement(self, dest):
        """Measure a qubit."""
        pass

    @abstractmethod
    def close_circuit(self):
        """Close the circuit. Many operation should be done here."""
        pass

    @abstractmethod
    def barrier(self):
        """Insert barrier."""
        pass