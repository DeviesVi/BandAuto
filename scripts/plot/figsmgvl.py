from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os

from matplotlib.ticker import ScalarFormatter
from defective_surface_code_adapter import Device, Analyzer
import pickle
import sinter
from typing import List
import numpy as np
from typing import List, Callable, Any, Dict, Tuple

def calculate_ler(sample: sinter.TaskStats):
    return sample.errors/(sample.shots - sample.discards)

notation_positions = (-0.25, 1)
notation_fontsize = 12
legend_fontsize = 8
marker_size = 4

plt.figure(figsize=(4, 3), dpi=100)

ax = plt.subplot(111)

device_dir = "manuscript_data/defective_devices/qubit_equal_coupler/device_d21_qdr0.02_cdr0.02/devices"
sample_dir_global = "manuscript_data/sample_data/samples/global_shell/samples_d21_qdr0.02_cdr0.02_p0.002"
sample_dir_local = "manuscript_data/sample_data/samples/local_shell/samples_d21_qdr0.02_cdr0.02_p0.002"

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
    global_samples_path = f"{sample_dir_global}/samples_{device.strong_id}.pkl"
    global_samples: List[sinter.TaskStats] = pickle.load(open(global_samples_path, "rb"))
    local_samples_path = f"{sample_dir_local}/samples_{device.strong_id}.pkl"
    local_samples: List[sinter.TaskStats] = pickle.load(open(local_samples_path, "rb"))
    samples = global_samples + local_samples

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

ax.step(*min_ler_relative_diff_cdf[("SPECIFIED", "LOCALAVG")]["0"], label="G-LA, $|0\\rangle$", color="C0", linewidth=1)
ax.step(*min_ler_relative_diff_cdf[("SPECIFIED", "LOCALAVG")]["+"], label="G-LA, $|+\\rangle$", color="C1", linewidth=1)
ax.step(*min_ler_relative_diff_cdf[("SPECIFIED", "LOCALMAX")]["0"], label="G-LM, $|0\\rangle$", color="C2", linewidth=1)
ax.step(*min_ler_relative_diff_cdf[("SPECIFIED", "LOCALMAX")]["+"], label="G-LM, $|+\\rangle$", color="C3", linewidth=1)
# Plot vertical line at median with x value on label
ax.axvline(x=np.median(min_ler_relative_diff[("SPECIFIED", "LOCALAVG")]["0"]), color="C0", linestyle="--", label=f"Med.: {np.median(min_ler_relative_diff[('SPECIFIED', 'LOCALAVG')]['0']):.2f}")
ax.axvline(x=np.median(min_ler_relative_diff[("SPECIFIED", "LOCALAVG")]["+"]), color="C1", linestyle="--", label=f"Med.: {np.median(min_ler_relative_diff[('SPECIFIED', 'LOCALAVG')]['+']):.2f}")
ax.axvline(x=np.median(min_ler_relative_diff[("SPECIFIED", "LOCALMAX")]["0"]), color="C2", linestyle="--", label=f"Med.: {np.median(min_ler_relative_diff[('SPECIFIED', 'LOCALMAX')]['0']):.2f}")
ax.axvline(x=np.median(min_ler_relative_diff[("SPECIFIED", "LOCALMAX")]["+"]), color="C3", linestyle="--", label=f"Med.: {np.median(min_ler_relative_diff[('SPECIFIED', 'LOCALMAX')]['+']):.2f}")

ax.legend(fontsize=legend_fontsize)
ax.set_ylabel("CDF")
ax.set_xlabel("Relative LER Difference")
ax.grid()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('figsmgvl.pdf', format='pdf')