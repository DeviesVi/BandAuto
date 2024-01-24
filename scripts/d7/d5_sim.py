from defective_surface_code_adapter import Device, SinterSampler, plot_graph, PhysicalErrors, HoldingCycleOption
import matplotlib.pyplot as plt
import pickle

device = Device(5,5)
sampler = SinterSampler(num_workers=8, max_shots=1000000, max_errors=1000000)
physical_errors = PhysicalErrors.SI1000_from_p(0.005)
specified_cycle = 1
holding_cycle_option = HoldingCycleOption.SPECIFIED
cycles = range(2,24,2)

plot_graph(device.graph,[])

def gen_tasks():
    print(f'Calculating for no defective qubit...')
    yield from sampler.gen_sinter_tasks(
        device=device,
        cycles=cycles,
        initial_states=["0"],
        physical_errors_list=[physical_errors],
        holding_cycle_option=holding_cycle_option,
        specified_holding_cycle=specified_cycle,
        metadata={'defective_qubit': None},
    )
    
def main():
    samples = sampler.sample(gen_tasks())
    pickle.dump(samples, open('samples_5.pkl', 'wb'))

if __name__ == '__main__':
    main()
    plt.show()