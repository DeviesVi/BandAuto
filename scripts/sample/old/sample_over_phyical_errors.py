import os
import numpy as np
import pickle
import json
from defective_surface_code_adapter import (
    SinterSampler, 
    Device, 
    HoldingCycleOption,
    PhysicalErrors,
)

source_dir = 'device_pool/device_d21_qdr0.02_cdr0.02/devices/'
destination_dir = 'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_p/'
selected_cycle_dir = 'device_pool/device_d21_qdr0.02_cdr0.02/cycle_selected_by_p0.002.json'

# Check destination directory exists
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Load selected cycles
selected_cycles = json.load(open(selected_cycle_dir, 'r'))

sampler = SinterSampler(num_workers=100)
holding_cycle_option = HoldingCycleOption.SPECIFIED

physical_errors_list = [PhysicalErrors.SI1000_from_p(p) for p in np.linspace(0.0005, 0.005, 10)]

def gen_tasks(device: Device):
        print(f'Generating samples for device {device.strong_id}')
        distance = min(device.data_width, device.data_height)
        specified_cycle = selected_cycles[device.strong_id]
        print(f'specified_cycle: {specified_cycle}')
        yield from sampler.gen_sinter_tasks(
            device=device,
            cycles=[distance],
            initial_states=["0","+"],
            physical_errors_list=physical_errors_list,
            holding_cycle_option=holding_cycle_option,
            specified_holding_cycle=specified_cycle,
        )

def main():
    for file in os.listdir(source_dir):
        device = Device.load(f'{source_dir}{file}')
        samples_path = f'{destination_dir}samples_{device.strong_id}.pkl'
        if os.path.exists(samples_path):
            print(f'Skipping device {device.strong_id} as samples already exist')
            continue
        try:
            samples = sampler.sample(gen_tasks(device))
        except:
            print(f'Failed to generate samples for device {device.strong_id}')
            # Log
            with open('failed_devices.txt', 'a') as f:
                f.write(f'{device.strong_id}\n')
            continue
        pickle.dump(samples, open(samples_path, 'wb'))


# NOTE: This is actually necessary! If the code inside 'main()' was at the
# module level, the multiprocessing children spawned by sinter.collect would
# also attempt to run that code.
if __name__ == "__main__":
    main()