import os
import numpy as np
import pickle
from defective_surface_code_adapter import (
    SinterSampler, 
    Device, 
    HoldingCycleOption,
    PhysicalErrors,
)

sampler = SinterSampler(num_workers=8)

def gen_tasks(device: Device, physical_errors: PhysicalErrors):
        print(f'Generating samples for device {device.strong_id}')
        distance = min(device.data_width, device.data_height)
        yield from sampler.gen_sinter_tasks(
            device=device,
            cycles=[distance],
            initial_states=["0","+"],
            physical_errors_list=[physical_errors],
            holding_cycle_option=HoldingCycleOption.SPECIFIED,
            specified_holding_cycle=1,
        )

def main():
    ds = [15, 21, 27]
    ps = [0.002, 0.003, 0.004]

    for d in ds:
        for p in ps:
            source_dir = f'manuscript_data/defective_devices/perfect/device_d{d}_qdr0_cdr0/devices'
            destination_dir = f'manuscript_data/sample_data/samples/perfect_ref/samples_d{d}_qdr0_cdr0_p{p}'
            physical_errors = PhysicalErrors.SI1000_from_p(p)
            for file in os.listdir(source_dir):
                device = Device.load(f'{source_dir}/{file}')
                samples_path = f'{destination_dir}/samples_{device.strong_id}.pkl'
                if os.path.exists(samples_path):
                    print(f'Skipping device {device.strong_id} as samples already exist')
                    continue
                try:
                    samples = sampler.sample(gen_tasks(device, physical_errors))
                    # Check destination directory exists
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    pickle.dump(samples, open(samples_path, 'wb'))
                except:
                    print(f'Failed to generate samples for device {device.strong_id}')
                    # Log
                    with open('failed_devices.txt', 'a') as f:
                        f.write(f'{device.strong_id}\n')
                    continue


# NOTE: This is actually necessary! If the code inside 'main()' was at the
# module level, the multiprocessing children spawned by sinter.collect would
# also attempt to run that code.
if __name__ == "__main__":
    main()