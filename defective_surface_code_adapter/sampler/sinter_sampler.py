from typing import List, Generator
from ..device import Device
from ..circuit_builder import StimBuilder
from ..circuit_builder.data import BuilderOptions, PhysicalErrors
import stim
import sinter


class SinterSampler:
    """Sinter sampler."""

    def __init__(
        self,
        max_shots: int = 1000000,
        max_errors: int = 1000,
        num_workers: int = 4,
        decoders: List[str] = ["pymatching"],
    ):
        """Constructor."""
        self._max_shots = max_shots
        self._max_errors = max_errors
        self._num_workers = num_workers
        self._decoders = decoders

    @staticmethod
    def gen_sinter_task(
        device: Device,
        cycle: int,
        initial_state: str,
        physical_errors: PhysicalErrors | None = None,
        metadata={},
    ) -> sinter.Task:
        options = BuilderOptions()
        options.physical_errors = (
            physical_errors if physical_errors is not None else PhysicalErrors()
        )
        builder = StimBuilder(device, options)
        circuit = builder.build(cycle, initial_state)
        task = sinter.Task(
            circuit=stim.Circuit(circuit),
            json_metadata={
                "cycle": cycle,
                "initial_state": initial_state,
                **metadata,
            },
        )
        return task

    def sample(self, tasks: List[sinter.Task]):
        """Sample."""
        samples = sinter.collect(
            num_workers=self._num_workers,
            max_shots=self._max_shots,
            max_errors=self._max_errors,
            tasks=tasks,
            decoders=["pymatching"],
        )
        return samples
