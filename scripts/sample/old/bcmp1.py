from typing import List
from defective_surface_code_adapter import Device, SinterSampler, plot_graph, PhysicalErrors, HoldingCycleOption
import matplotlib.pyplot as plt
import pickle
import sinter

sampler = SinterSampler(num_workers=8, max_shots=1000000, max_errors=1000000)
physical_errors = PhysicalErrors.SI1000_from_p(0.002)
specified_cycles = range(1,7)
holding_cycle_option = HoldingCycleOption.SPECIFIED


def gen_tasks(devices: List[Device]):
    for i, device in enumerate(devices):
        print(f'Calculating for device {device.strong_id}...')
        for specified_cycle in specified_cycles:
            print(f'Specified cycle {specified_cycle}...')
            yield from sampler.gen_sinter_tasks(
                device=device,
                cycles=[7],
                initial_states=["0","+"],
                physical_errors_list=[physical_errors],
                holding_cycle_option=holding_cycle_option,
                specified_holding_cycle=specified_cycle,
                metadata={'device_index': i},
            )

def main():
    device0 = Device(7,7)
    device0.add_ractangle_defect(7,7,1,1, clear_defect=False)

    device1 = Device(7,7)
    device1.add_ractangle_defect(7,7,1,1, clear_defect=False)

    samples = sampler.sample(gen_tasks([device0, device1]))

    pickle.dump(samples, open('sp_data/bcmp1.pkl', 'wb'))


if __name__ == '__main__':
    main()
