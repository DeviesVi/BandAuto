import matplotlib.pyplot as plt
import os
from defective_surface_code_adapter import Device, Analyzer
import pickle
import sinter
from typing import List, Callable, Any, Dict, Tuple
import numpy as np
from data_dir import device_dir, sample_dir

min_ler_pos = {
    "SPEC": {"0": [], "+": []},
    "AVG": {"0": [], "+": []},
    "MAX": {"0": [], "+": []},
}

filter_funcs: Dict[Tuple[str, str], Callable[['sinter.TaskStats'], Any]] = {
    ("SPEC", "0"): lambda stat: stat.json_metadata["initial_state"] == "0" and stat.json_metadata["holding_cycle_option"] == "SPEC",
    ("SPEC", "+"): lambda stat: stat.json_metadata["initial_state"] == "+" and stat.json_metadata["holding_cycle_option"] == "SPEC",
    ("AVG", "0"): lambda stat: stat.json_metadata["initial_state"] == "0" and stat.json_metadata["holding_cycle_option"] == "AVG",
    ("AVG", "+"): lambda stat: stat.json_metadata["initial_state"] == "+" and stat.json_metadata["holding_cycle_option"] == "AVG",
    ("MAX", "0"): lambda stat: stat.json_metadata["initial_state"] == "0" and stat.json_metadata["holding_cycle_option"] == "MAX",
    ("MAX", "+"): lambda stat: stat.json_metadata["initial_state"] == "+" and stat.json_metadata["holding_cycle_option"] == "MAX",
}

def calculate_ler(sample: sinter.TaskStats):
    return sample.errors/(sample.shots - sample.discards)

def get_min_ler_pos(device:Device, samples: List[sinter.TaskStats], filter_func: Callable[['sinter.TaskStats'], Any] = lambda _: True):
    # Record the position of the minimum LER, according to specified_holding_cycle or holding_cycle_ratio
    result = Analyzer.analyze_device(device)
    avg_weight = result.stabilizer_statistics['avg_stabilizer_weight']

    min_ler_pos = None
    for sample in samples:
        if filter_func(sample):
            ler = calculate_ler(sample)
            if min_ler_pos is None:
                min_ler_pos = sample.json_metadata["specified_holding_cycle"]/avg_weight if sample.json_metadata["holding_cycle_option"] == "SPEC" else sample.json_metadata["holding_cycle_ratio"]
                min_ler = ler
            else:
                if ler < min_ler:
                    min_ler_pos = sample.json_metadata["specified_holding_cycle"]/avg_weight if sample.json_metadata["holding_cycle_option"] == "SPEC" else sample.json_metadata["holding_cycle_ratio"]
                    min_ler = ler
    if min_ler_pos < 0.1:
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
    device = Device.load(f"{device_dir}{file}")
    samples_path = f"{sample_dir}samples_{device.strong_id}.pkl"
    samples: List[sinter.TaskStats] = pickle.load(open(samples_path, "rb"))

    for (holding_cycle_option, initial_state), filter_func in filter_funcs.items():
        min_ler_pos[holding_cycle_option][initial_state].append(get_min_ler_pos(device, samples, filter_func))

min_ler_pos_cdf = {
    "SPEC": {
        "0": calculate_cdf(min_ler_pos["SPEC"]["0"]),
        "+": calculate_cdf(min_ler_pos["SPEC"]["+"]),
    },
    "AVG": {
        "0": calculate_cdf(min_ler_pos["AVG"]["0"]),
        "+": calculate_cdf(min_ler_pos["AVG"]["+"]),
    },
    "MAX": {
        "0": calculate_cdf(min_ler_pos["MAX"]["0"]),
        "+": calculate_cdf(min_ler_pos["MAX"]["+"]),
    },
}

ax = plt.subplot(121)
# Plot SPEC
ax.step(min_ler_pos_cdf["SPEC"]["0"][0], min_ler_pos_cdf["SPEC"]["0"][1], label="SPEC, 0", linewidth=2, color="blue")
ax.step(min_ler_pos_cdf["SPEC"]["+"][0], min_ler_pos_cdf["SPEC"]["+"][1], label="SPEC, +", linewidth=2, color="orange")
# Plot vertical line at median
ax.axvline(np.median(min_ler_pos["SPEC"]["0"]), color="blue", linestyle="--", label=f"Median: {np.median(min_ler_pos['SPEC']['0']):.2f}")
ax.axvline(np.median(min_ler_pos["SPEC"]["+"]), color="orange", linestyle="--", label=f"Median: {np.median(min_ler_pos['SPEC']['+']):.2f}")


ax.legend()
ax.grid()
ax.set_title("CDF of Minimum LER Position")
ax.set_xlabel("N Shell/Average Weight")
ax.set_ylabel("CDF")
ax.set_ylim(0, 1)

# Plot AVG and MAX
ax = plt.subplot(122)
# Plot AVG
ax.step(min_ler_pos_cdf["AVG"]["0"][0], min_ler_pos_cdf["AVG"]["0"][1], label="AVG, 0", linewidth=2, color="blue")
ax.step(min_ler_pos_cdf["AVG"]["+"][0], min_ler_pos_cdf["AVG"]["+"][1], label="AVG, +", linewidth=2, color="orange")
# Plot MAX
ax.step(min_ler_pos_cdf["MAX"]["0"][0], min_ler_pos_cdf["MAX"]["0"][1], label="MAX, 0", linewidth=2, color="green")
ax.step(min_ler_pos_cdf["MAX"]["+"][0], min_ler_pos_cdf["MAX"]["+"][1], label="MAX, +", linewidth=2, color="red")
# Plot vertical line at median
ax.axvline(np.median(min_ler_pos["AVG"]["0"]), color="blue", linestyle="--", label=f"Median: {np.median(min_ler_pos['AVG']['0']):.2f}")
ax.axvline(np.median(min_ler_pos["AVG"]["+"]), color="orange", linestyle="--", label=f"Median: {np.median(min_ler_pos['AVG']['+']):.2f}")
ax.axvline(np.median(min_ler_pos["MAX"]["0"]), color="green", linestyle="dashdot", label=f"Median: {np.median(min_ler_pos['MAX']['0']):.2f}")
ax.axvline(np.median(min_ler_pos["MAX"]["+"]), color="red", linestyle="dashdot", label=f"Median: {np.median(min_ler_pos['MAX']['+']):.2f}")
ax.legend()
ax.grid()
ax.set_title("CDF of Minimum LER Position")
ax.set_xlabel("Holding Cycle Ratio")
ax.set_ylabel("CDF")
ax.set_ylim(0, 1)

plt.show()