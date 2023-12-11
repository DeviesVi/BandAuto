from typing import Any, Dict, List, Generator
from ..device import Device
from ..circuit_builder import StimBuilder
from ..circuit_builder.data import BuilderOptions, PhysicalErrors, HoldingCycleOption
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
    def gen_sinter_tasks(
        device: Device,
        cycles: List[int],
        initial_states: List[str],
        physical_errors_list: List[PhysicalErrors],
        holding_cycle_option: HoldingCycleOption = HoldingCycleOption.GLOBALAVG,
        holding_cycle_ratio: float | None = None,
        holding_cycle_ratio_x: float | None = None,
        holding_cycle_ratio_z: float | None = None,
        specified_holding_cycle_x: int | None = None,
        specified_holding_cycle_z: int | None = None,
        metadata: Dict[str, Any] = {},
    ) -> Generator[sinter.Task, None, None]:
        for physical_errors in physical_errors_list:
            options = BuilderOptions()
            options.physical_errors = physical_errors
            options.stabilizer_group_holding_cycle_option = holding_cycle_option
            options.stabilizer_group_holding_cycle_ratio = holding_cycle_ratio
            options.stabilizer_group_holding_cycle_ratio_x = holding_cycle_ratio_x
            options.stabilizer_group_holding_cycle_ratio_z = holding_cycle_ratio_z
            options.stabilizer_group_specified_holding_cycle_x = specified_holding_cycle_x
            options.stabilizer_group_specified_holding_cycle_z = specified_holding_cycle_z

            if holding_cycle_option == HoldingCycleOption.SPEC:
                assert specified_holding_cycle_x is not None
                assert specified_holding_cycle_z is not None
            else:
                assert holding_cycle_ratio is not None or holding_cycle_ratio_x is not None and holding_cycle_ratio_z is not None

            builder = StimBuilder(device, options)
            for initial_state in initial_states:
                for cycle in cycles:
                    circuit = builder.build(cycle, initial_state)
                    yield sinter.Task(
                        circuit=stim.Circuit(circuit),
                        json_metadata={
                            "device": device.strong_id,
                            "cycle": cycle,
                            "initial_state": initial_state,
                            "physical_errors": physical_errors.to_dict(),
                            **metadata,
                        },
                    )

    def sample(self, tasks: Generator[sinter.Task, None, None]) -> List[sinter.TaskStats]:
        """Do sampling.
        Use if __name__ == '__main__' to avoid multiprocessing issues.
        """
        samples = sinter.collect(
            num_workers=self._num_workers,
            max_shots=self._max_shots,
            max_errors=self._max_errors,
            tasks=tasks,
            decoders=["pymatching"],
        )
        return samples
