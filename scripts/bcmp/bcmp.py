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
    device0.add_ractangle_defect(9,9,1,1, clear_defect=False)

    device1 = Device(7,7)
    device1.add_ractangle_defect(7,7,1,1, clear_defect=False)
    device1.add_ractangle_defect(9,9,1,1, clear_defect=False)
    device1.add_ractangle_defect(8,8,1,1, clear_defect=False)

    legend = {
        0: 'Bandage',
        1: 'Trandition',
    }

    samples = sampler.sample(gen_tasks([device0, device1]))

    plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(111)
    sinter.plot_error_rate(
        ax=ax,
        stats = samples,
        group_func=lambda stat: f'{legend[stat.json_metadata["device_index"]]}|{stat.json_metadata["initial_state"]}>',
        x_func=lambda stat: stat.json_metadata["specified_holding_cycle"],
        highlight_max_likelihood_factor=1,
        plot_args_func=lambda index, curve_id: {'marker': None},
    )
    ax.grid()
    ax.set_title("LER vs Global Shell Size")
    ax.set_ylabel("LER")
    ax.set_xlabel("Global Shell Size")
    ax.legend()
    plt.savefig('bcmp.pdf', format='pdf')
    plt.show()


if __name__ == '__main__':
    main()
