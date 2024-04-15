import matplotlib.pyplot as plt
import os
from defective_surface_code_adapter import Device, Analyzer
import pickle
import sinter
from typing import List, Callable, Any, Dict, Tuple
import numpy as np

device_dir = "device_pool/device_d21_qdr0.02_cdr0.02/devices"
sample_dir = "device_pool/device_d21_qdr0.02_cdr0.02/samples_over_holding_options_p0.002"


min_ler = {
    "SPECIFIED": {"0": [], "+": []},
    "LOCALAVG": {"0": [], "+": []},
    "LOCALMAX": {"0": [], "+": []},
}

filter_funcs: Dict[Tuple[str, str], Callable[['sinter.TaskStats'], Any]] = {
    ("SPECIFIED", "0"): lambda stat: stat.json_metadata["initial_state"] == "0" and stat.json_metadata["holding_cycle_option"] == "SPECIFIED",
    ("SPECIFIED", "+"): lambda stat: stat.json_metadata["initial_state"] == "+" and stat.json_metadata["holding_cycle_option"] == "SPECIFIED",
    ("LOCALAVG", "0"): lambda stat: stat.json_metadata["initial_state"] == "0" and stat.json_metadata["holding_cycle_option"] == "LOCALAVG",
    ("LOCALAVG", "+"): lambda stat: stat.json_metadata["initial_state"] == "+" and stat.json_metadata["holding_cycle_option"] == "LOCALAVG",
    ("LOCALMAX", "0"): lambda stat: stat.json_metadata["initial_state"] == "0" and stat.json_metadata["holding_cycle_option"] == "LOCALMAX",
    ("LOCALMAX", "+"): lambda stat: stat.json_metadata["initial_state"] == "+" and stat.json_metadata["holding_cycle_option"] == "LOCALMAX",
}

def calculate_ler(sample: sinter.TaskStats):
    return sample.errors/(sample.shots - sample.discards)

def get_min_ler(samples: List[sinter.TaskStats], filter_func: Callable[['sinter.TaskStats'], Any] = lambda _: True):
    ler = []
    for sample in samples:
        if filter_func(sample):
            ler.append(calculate_ler(sample))
    return min(ler)

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
        min_ler[holding_cycle_option][initial_state].append(get_min_ler(samples, filter_func))

min_ler_relative_diff = {
    ("SPECIFIED", "LOCALAVG"): {
        "0": [(a - b) / b for a, b in zip(min_ler["SPECIFIED"]["0"], min_ler["LOCALAVG"]["0"])],
        "+": [(a - b) / b for a, b in zip(min_ler["SPECIFIED"]["+"], min_ler["LOCALAVG"]["+"])],
    },
    ("SPECIFIED", "LOCALMAX"): {
        "0": [(a - b) / b for a, b in zip(min_ler["SPECIFIED"]["0"], min_ler["LOCALMAX"]["0"])],
        "+": [(a - b) / b for a, b in zip(min_ler["SPECIFIED"]["+"], min_ler["LOCALMAX"]["+"])],
    },
}

min_ler_relative_diff_cdf = {
    ("SPECIFIED", "LOCALAVG"): {
        "0": calculate_cdf(min_ler_relative_diff[("SPECIFIED", "LOCALAVG")]["0"]),
        "+": calculate_cdf(min_ler_relative_diff[("SPECIFIED", "LOCALAVG")]["+"]),
    },
    ("SPECIFIED", "LOCALMAX"): {
        "0": calculate_cdf(min_ler_relative_diff[("SPECIFIED", "LOCALMAX")]["0"]),
        "+": calculate_cdf(min_ler_relative_diff[("SPECIFIED", "LOCALMAX")]["+"]),
    },
}

# Plot CDF using plt.step
fig, ax = plt.subplots()
ax.step(*min_ler_relative_diff_cdf[("SPECIFIED", "LOCALAVG")]["0"], label="(SPEC-AVG)/AVG, 0", color="blue", linewidth=2)
ax.step(*min_ler_relative_diff_cdf[("SPECIFIED", "LOCALAVG")]["+"], label="(SPEC-AVG)/AVG, +", color="orange", linewidth=2)
ax.step(*min_ler_relative_diff_cdf[("SPECIFIED", "LOCALMAX")]["0"], label="(SPEC-MAX)/MAX, 0", color="green", linewidth=2)
ax.step(*min_ler_relative_diff_cdf[("SPECIFIED", "LOCALMAX")]["+"], label="(SPEC-MAX)/MAX, +", color="red", linewidth=2)
# Plot vertical line at median with x value on label
ax.axvline(x=np.median(min_ler_relative_diff[("SPECIFIED", "LOCALAVG")]["0"]), color="blue", linestyle="--", label=f"Median: {np.median(min_ler_relative_diff[('SPECIFIED', 'LOCALAVG')]['0']):.2f}")
ax.axvline(x=np.median(min_ler_relative_diff[("SPECIFIED", "LOCALAVG")]["+"]), color="orange", linestyle="--", label=f"Median: {np.median(min_ler_relative_diff[('SPECIFIED', 'LOCALAVG')]['+']):.2f}")
ax.axvline(x=np.median(min_ler_relative_diff[("SPECIFIED", "LOCALMAX")]["0"]), color="green", linestyle="--", label=f"Median: {np.median(min_ler_relative_diff[('SPECIFIED', 'LOCALMAX')]['0']):.2f}")
ax.axvline(x=np.median(min_ler_relative_diff[("SPECIFIED", "LOCALMAX")]["+"]), color="red", linestyle="--", label=f"Median: {np.median(min_ler_relative_diff[('SPECIFIED', 'LOCALMAX')]['+']):.2f}")

ax.legend()
ax.set_title("CDF of Relative Difference of Min LER")
ax.set_ylabel("CDF")
ax.set_xlabel("Relative Difference of Min LER")
ax.grid()
ax.set_ylim(0, 1)

plt.show()