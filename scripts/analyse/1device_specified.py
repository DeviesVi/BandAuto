import matplotlib.pyplot as plt
import os
from defective_surface_code_adapter import Device, Analyzer, Adapter, plot_graph
import pickle
import sinter
from typing import List
import numpy as np

min_ler_s_list = []

device_dir = 'device_pool/device_d21_qdr0.005_cdr0.005/devices/device_102f51cd950d9cad7d71545c685daac26e3264c852bf0ee4258ed3d56cec8d7c.pkl'
sample_dir = 'device_pool/device_d21_qdr0.005_cdr0.005/samples_over_specified_cycle'


device = Device.load(f"{device_dir}")
print(device.strong_id)
samples_path = f"{sample_dir}/samples_{device.strong_id}.pkl"
samples: List[sinter.TaskStats] = pickle.load(open(samples_path, "rb"))

# Classify samples according to holding cycle option and initial state
spec_samples_zero: List[sinter.TaskStats] = []
spec_samples_plus: List[sinter.TaskStats] = []
for sample in samples:
    if sample.json_metadata["initial_state"] == "0":
        spec_samples_zero.append(sample)
    elif sample.json_metadata["initial_state"] == "+":
        spec_samples_plus.append(sample)

ax = plt.subplot(121)

sinter.plot_error_rate(
    ax=ax,
    stats=spec_samples_zero + spec_samples_plus,
    group_func=lambda stat: stat.json_metadata["initial_state"],
    x_func=lambda stat: stat.json_metadata["specified_holding_cycle"],
)

ax.grid()
ax.set_title("Logical Error Rate vs N Shell")
ax.set_ylabel("Logical Error Probability (per shot)")
ax.set_xlabel("N Shell")
ax.legend()

ax = plt.subplot(122)
adapter_result = Adapter.adapt_device(device)
plot_graph(device.graph, adapter_result.disabled_nodes)

analyzer_result = Analyzer.analyze_device(device)
print(analyzer_result)

plt.show()