import sinter
from defective_surface_code_adapter import Device, SinterSampler, PhysicalErrors, HoldingCycleOption
from typing import List
import numpy as np
import pickle

device_path = 'device_pool/device_d21_qdr0.02_cdr0.02/devices/device_f18efd2f3ded1a49e8011fd565ecb19bf277c3010172993707f78c633d2dac40.pkl'

device = Device.load(device_path)
sampler = SinterSampler(num_workers=8)
d = min(device.data_width, device.data_height)
specified_cycles = range(1, (d+1)//2)

physical_errors_list = [PhysicalErrors.SI1000_from_p(p) for p in np.linspace(0.001, 0.005, 5)]

def gen_tasks():
    print(f'Calculating for device {device.strong_id}...')
    for specified_cycle in specified_cycles:
        print(f'Specified cycle {specified_cycle}...')
        yield from sampler.gen_sinter_tasks(
            device=device,
            cycles=[7],
            initial_states=["0","+"],
            physical_errors_list=physical_errors_list,
            holding_cycle_option=HoldingCycleOption.SPECIFIED,
            specified_holding_cycle=specified_cycle,
        )

def main():
    samples = sampler.sample(gen_tasks())
    pickle.dump(samples, open('sp_data/nshell_p_cor.pkl', 'wb'))

if __name__ == '__main__':
    main()