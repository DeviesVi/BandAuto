import matplotlib.pyplot as plt
import os
from defective_surface_code_adapter import Device, Analyzer, Adapter, plot_graph
import pickle
import sinter
from typing import List
import numpy as np
from data_dir import device_dir, sample_dir

min_ler_s_list = []
min_ler_m_list = []
min_ler_a_list = []

device = "0927d9540e635e9813aa291684e4ec9f9726db5776d5100d886620501ca7baac"

print(device)

device = Device.load(f"{device_dir}device_{device}.pkl")
samples_path = f"{sample_dir}/samples_{device.strong_id}.pkl"
samples: List[sinter.TaskStats] = pickle.load(open(samples_path, "rb"))

# Classify samples according to holding cycle option and initial state
max_samples_zero: List[sinter.TaskStats] = []
max_samples_plus: List[sinter.TaskStats] = []
avg_samples_zero: List[sinter.TaskStats] = []
avg_samples_plus: List[sinter.TaskStats] = []
spec_samples_zero: List[sinter.TaskStats] = []
spec_samples_plus: List[sinter.TaskStats] = []
for sample in samples:
    if sample.json_metadata["holding_cycle_option"] == "LOCALMAX":
        if sample.json_metadata["initial_state"] == "0":
            max_samples_zero.append(sample)
        elif sample.json_metadata["initial_state"] == "+":
            max_samples_plus.append(sample)
    elif sample.json_metadata["holding_cycle_option"] == "LOCALAVG":
        if sample.json_metadata["initial_state"] == "0":
            avg_samples_zero.append(sample)
        elif sample.json_metadata["initial_state"] == "+":
            avg_samples_plus.append(sample)
    elif sample.json_metadata["holding_cycle_option"] == "SPECIFIED":
        if sample.json_metadata["initial_state"] == "0":
            spec_samples_zero.append(sample)
        elif sample.json_metadata["initial_state"] == "+":
            spec_samples_plus.append(sample)

ax = plt.subplot(221)

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


ax = plt.subplot(223)

sinter.plot_error_rate(
    ax=ax,
    stats=avg_samples_zero + avg_samples_plus + max_samples_zero + max_samples_plus,
    group_func=lambda stat: f'{stat.json_metadata["holding_cycle_option"]}, {stat.json_metadata["initial_state"]}',
    x_func=lambda stat: stat.json_metadata["holding_cycle_ratio"],
)

ax.grid()
ax.set_title("Logical Error Rate vs Shell Ratio")
ax.set_ylabel("Logical Error Probability (per shot)")
ax.set_xlabel("Shell Ratio")
ax.legend()

ax = plt.subplot(122)
adapter_result = Adapter.adapt_device(device)
plot_graph(device.graph, adapter_result.disabled_nodes)

analyzer_result = Analyzer.analyze_device(device)
print(analyzer_result)

plt.show()