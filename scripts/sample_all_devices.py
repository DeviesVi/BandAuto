import os
import numpy as np
import pickle
from defective_surface_code_adapter import (
    SinterSampler, 
    Device, 
    HoldingCycleOption,
    PhysicalErrors,
)

source_dir = 'device_pool/devices/'
destination_dir = 'device_pool/samples/'

# Check destination directory exists
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

sampler = SinterSampler(num_workers=8)
holding_cycle_options = [HoldingCycleOption.MAX, HoldingCycleOption.AVG, HoldingCycleOption.SPEC]

physical_errors = PhysicalErrors.SI1000_from_p(0.002)
ratio_range = np.linspace(0.02, 0.80, 40)
specified_cycle_range = range(1, 11)

def gen_tasks(device: Device):
        print(f'Generating samples for device {device.strong_id}')
        distace = min(device.data_width, device.data_height)
        for holding_cycle_option in holding_cycle_options:
            print(f'holding_cycle_option: {holding_cycle_option.name}')
            if holding_cycle_option == HoldingCycleOption.SPEC:
                for specified_cycle in specified_cycle_range:
                    print(f'specified_cycle: {specified_cycle}')
                    yield from sampler.gen_sinter_tasks(
                        device=device,
                        cycles=[distace],
                        initial_states=["0","+"],
                        physical_errors_list=[physical_errors],
                        holding_cycle_option=holding_cycle_option,
                        specified_holding_cycle=specified_cycle,
                    )
            else:
                for ratio in ratio_range:
                    print(f'ratio: {ratio}')
                    yield from sampler.gen_sinter_tasks(
                        device=device,
                        cycles=[distace],
                        initial_states=["0","+"],
                        physical_errors_list=[physical_errors],
                        holding_cycle_option=holding_cycle_option,
                        holding_cycle_ratio=ratio,
                    )

def main():
    for file in os.listdir(source_dir):
        device = Device.load(f'{source_dir}{file}')
        samples = sampler.sample(gen_tasks(device))
        pickle.dump(samples, open(f'{destination_dir}samples_{device.strong_id}.pkl', 'wb'))


# NOTE: This is actually necessary! If the code inside 'main()' was at the
# module level, the multiprocessing children spawned by sinter.collect would
# also attempt to run that code.
if __name__ == "__main__":
    main()