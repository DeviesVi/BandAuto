import matplotlib.pyplot as plt
import os
from defective_surface_code_adapter import Device, Analyzer
import pickle
import sinter
from typing import List, Callable, Any, Dict, Tuple
import numpy as np
from utils import calculate_ler

device_dir = "device_pool/device_d21_qdr0.005_cdr0.005/devices"
sample_dir = "device_pool/device_d21_qdr0.005_cdr0.005/samples_over_specified_cycle"

min_ler_pos = {
    "SPECIFIED": {"0": [], "+": []},
}

filter_funcs: Dict[Tuple[str, str], Callable[['sinter.TaskStats'], Any]] = {
    ("SPECIFIED", "0"): lambda stat: stat.json_metadata["initial_state"] == "0" and stat.json_metadata["holding_cycle_option"] == "SPECIFIED",
    ("SPECIFIED", "+"): lambda stat: stat.json_metadata["initial_state"] == "+" and stat.json_metadata["holding_cycle_option"] == "SPECIFIED",
}

def get_min_ler_pos(device:Device, samples: List[sinter.TaskStats], filter_func: Callable[['sinter.TaskStats'], Any] = lambda _: True, devide_weight = True):
    # Record the position of the minimum LER, according to specified_holding_cycle or holding_cycle_ratio
    result = Analyzer.analyze_device(device)
    avg_weight = result.stabilizer_statistics['avg_stabilizer_weight']

    min_ler_pos = None
    for sample in samples:
        if filter_func(sample):
            ler = calculate_ler(sample)
            if min_ler_pos is None:
                if devide_weight:
                    min_ler_pos = sample.json_metadata["specified_holding_cycle"]/avg_weight if sample.json_metadata["holding_cycle_option"] == "SPECIFIED" else sample.json_metadata["holding_cycle_ratio"]
                else:
                    min_ler_pos = sample.json_metadata["specified_holding_cycle"] if sample.json_metadata["holding_cycle_option"] == "SPECIFIED" else sample.json_metadata["holding_cycle_ratio"]
                min_ler = ler
            else:
                if ler < min_ler:
                    if devide_weight:
                        min_ler_pos = sample.json_metadata["specified_holding_cycle"]/avg_weight if sample.json_metadata["holding_cycle_option"] == "SPECIFIED" else sample.json_metadata["holding_cycle_ratio"]
                    else:
                        min_ler_pos = sample.json_metadata["specified_holding_cycle"] if sample.json_metadata["holding_cycle_option"] == "SPECIFIED" else sample.json_metadata["holding_cycle_ratio"]
                    min_ler = ler
    if min_ler_pos > 0.7:
        print(device.strong_id)
    return min_ler_pos

def calculate_cdf(data):
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    # Add 0 to beginning of CDF
    cdf = np.insert(cdf, 0, 0)
    # Add first data point to beginning of data
    data_sorted = np.insert(data_sorted, 0, data_sorted[0])
    return data_sorted, cdf

for file in os.listdir(device_dir):
    device = Device.load(f"{device_dir}/{file}")
    samples_path = f"{sample_dir}/samples_{device.strong_id}.pkl"
    samples: List[sinter.TaskStats] = pickle.load(open(samples_path, "rb"))

    for (holding_cycle_option, initial_state), filter_func in filter_funcs.items():
        mlp = get_min_ler_pos(device, samples, filter_func)
        min_ler_pos[holding_cycle_option][initial_state].append(mlp)
        

min_ler_pos_cdf = {
    "SPECIFIED": {
        "0": calculate_cdf(min_ler_pos["SPECIFIED"]["0"]),
        "+": calculate_cdf(min_ler_pos["SPECIFIED"]["+"]),
    },
}

ax = plt.subplot(111)
# Plot SPECIFIED
ax.step(min_ler_pos_cdf["SPECIFIED"]["0"][0], min_ler_pos_cdf["SPECIFIED"]["0"][1], label="SPECIFIED, 0", linewidth=2, color="blue")
ax.step(min_ler_pos_cdf["SPECIFIED"]["+"][0], min_ler_pos_cdf["SPECIFIED"]["+"][1], label="SPECIFIED, +", linewidth=2, color="orange")
# Plot vertical line at median
ax.axvline(np.median(min_ler_pos["SPECIFIED"]["0"]), color="blue", linestyle="--", label=f"Median: {np.median(min_ler_pos['SPECIFIED']['0']):.2f}")
ax.axvline(np.median(min_ler_pos["SPECIFIED"]["+"]), color="orange", linestyle="--", label=f"Median: {np.median(min_ler_pos['SPECIFIED']['+']):.2f}")


ax.legend()
ax.grid()
ax.set_title("CDF of Minimum LER Position")
ax.set_xlabel("N Shell/Average Weight")
ax.set_ylabel("CDF")
ax.set_ylim(0, 1)

plt.show()