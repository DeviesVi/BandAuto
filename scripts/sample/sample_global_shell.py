import os
import pickle
from defective_surface_code_adapter import (
    SinterSampler, 
    Device, 
    HoldingCycleOption,
    PhysicalErrors,
)

sampler = SinterSampler(num_workers=100)

def gen_tasks(device: Device, shell_size_range: range, physical_errors: PhysicalErrors):
        print(f'Generating samples for device {device.strong_id}')
        distance = min(device.data_width, device.data_height)
        for shell_size in shell_size_range:
            print(f'specified_cycle: {shell_size}')
            yield from sampler.gen_sinter_tasks(
                device=device,
                cycles=[distance],
                initial_states=["0","+"],
                physical_errors_list=[physical_errors],
                holding_cycle_option=HoldingCycleOption.SPECIFIED,
                specified_holding_cycle=shell_size,
            )

def main():
    ds = [15, 21, 27]
    defect_rates = [0.005, 0.01, 0.015, 0.02]

    p = 0.002
    physical_errors = PhysicalErrors.SI1000_from_p(p)

    for d in ds:
        shell_size_range = range(1, (d+1)//2)
        for dr in defect_rates:
            source_dir = f'manuscript_data/defective_devices/qubit_equal_coupler/device_d{d}_qdr{dr}_cdr{dr}/devices'
            destination_dir = f'manuscript_data/sample_data/samples/global_shell/samples_d{d}_qdr{dr}_cdr{dr}_p{p}'

            for file in os.listdir(source_dir):
                device = Device.load(f'{source_dir}/{file}')
                samples_path = f'{destination_dir}/samples_{device.strong_id}.pkl'
                if os.path.exists(samples_path):
                    print(f'Skipping device {device.strong_id} as samples already exist')
                    continue
                try:
                    samples = sampler.sample(gen_tasks(device, shell_size_range, physical_errors))
                    # Check destination directory exists
                    if not os.path.exists(destination_dir):
                        os.makedirs(destination_dir)
                    pickle.dump(samples, open(samples_path, 'wb'))
                except Exception as e:
                    print(f'Failed to generate samples for device {device.strong_id}')
                    print(e)
                    # Log
                    with open('failed_devices.txt', 'a') as f:
                        f.write(f'{device.strong_id}\n')
                    continue


# NOTE: This is actually necessary! If the code inside 'main()' was at the
# module level, the multiprocessing children spawned by sinter.collect would
# also attempt to run that code.
if __name__ == "__main__":
    main()