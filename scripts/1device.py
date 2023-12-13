import matplotlib.pyplot as plt
import os
from defective_surface_code_adapter import Device, Analyzer
import pickle
import sinter
from typing import List
import numpy as np

device_path = "device_pool\devices\device_eb47b63d049950f90bac30f696de2db5136b46147803ae34d77c0a81da773048.pkl"

min_ler_s_list = []
min_ler_m_list = []
min_ler_a_list = []


device = Device.load(device_path)
samples_path = f"device_pool/samples/samples_{device.strong_id}.pkl"
samples: List[sinter.TaskStats] = pickle.load(open(samples_path, "rb"))

# Classify samples according to holding cycle option and initial state
max_samples_zero: List[sinter.TaskStats] = []
max_samples_plus: List[sinter.TaskStats] = []
avg_samples_zero: List[sinter.TaskStats] = []
avg_samples_plus: List[sinter.TaskStats] = []
spec_samples_zero: List[sinter.TaskStats] = []
spec_samples_plus: List[sinter.TaskStats] = []
for sample in samples:
    if sample.json_metadata["holding_cycle_option"] == "MAX":
        if sample.json_metadata["initial_state"] == "0":
            max_samples_zero.append(sample)
        elif sample.json_metadata["initial_state"] == "+":
            max_samples_plus.append(sample)
    elif sample.json_metadata["holding_cycle_option"] == "AVG":
        if sample.json_metadata["initial_state"] == "0":
            avg_samples_zero.append(sample)
        elif sample.json_metadata["initial_state"] == "+":
            avg_samples_plus.append(sample)
    elif sample.json_metadata["holding_cycle_option"] == "SPEC":
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

result = Analyzer.analyze_device(device)
print(result)

plt.show()