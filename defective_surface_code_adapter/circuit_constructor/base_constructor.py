"""Base constructor to build circuits for the surface code."""

from abc import ABC, abstractmethod

from ..device import Device
from .data import ConstructorOptions
from ..adapter import Adapter

class BaseConstructor(ABC):
    """Base constructor to build circuits for the surface code."""

    def __init__(self, device:Device, options: ConstructorOptions | None = None) -> None:
        """Constructor."""
        self.device = device

        if options is None:
            self.options = ConstructorOptions()
        else:
            self.options = options
        

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
    def barrier(self):
        """Insert barrier."""
        pass