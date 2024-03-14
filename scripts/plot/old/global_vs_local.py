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

plt.figure(figsize=(8, 6), dpi=100)

# fig a
print('Making fig a...')
ax = plt.subplot(221)

device_path = 'device_pool/device_d21_qdr0.02_cdr0.02/devices/device_f18efd2f3ded1a49e8011fd565ecb19bf277c3010172993707f78c633d2dac40.pkl'
sample_dir = 'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_holding_options_p0.002'

device = Device.load(device_path)
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

sinter.plot_error_rate(
    ax=ax,
    stats=spec_samples_zero + spec_samples_plus,
    group_func=lambda stat: f'G, $|{stat.json_metadata["initial_state"]}\\rangle$',
    x_func=lambda stat: stat.json_metadata["specified_holding_cycle"],
    highlight_max_likelihood_factor=1,
    plot_args_func=lambda index, curve_id: {
        'marker': 'o' if index == 0 else '^',
        'markersize': marker_size,
        'color': 'C0' if index == 0 else 'C2'
    },
)

ax.grid()
ax.set_ylabel("LER")
ax.set_xlabel("Global Shell Size")
ax.legend(fontsize=legend_fontsize)
ax.text(*notation_positions, '(a)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# fig b
print('Making fig b...')
ax = plt.subplot(223)

sinter.plot_error_rate(
    ax=ax,
    stats=avg_samples_zero + avg_samples_plus + max_samples_zero + max_samples_plus,
    group_func=lambda stat: f'L{stat.json_metadata["holding_cycle_option"][5]}, $|{stat.json_metadata["initial_state"]}\\rangle$',
    x_func=lambda stat: stat.json_metadata["holding_cycle_ratio"],
    highlight_max_likelihood_factor=1,
    plot_args_func=lambda index, curve_id: {'marker': None},
)

ax.grid()
ax.set_ylim(0.0075, 0.040)
ax.set_ylabel("LER")
ax.set_xlabel("Local Shell Ratio")
ax.legend(fontsize=legend_fontsize)
ax.text(*notation_positions, '(b)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# fig c
print('Making fig c...')
ax = plt.subplot(222)

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
ax.set_xlabel("Relative LER Diff at Sweet Point")
ax.grid()
ax.set_ylim(0, 1)
ax.text(*notation_positions, '(c)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# fig 2d
print('Making fig d...')
ax = plt.subplot(224)

device_dir = "device_pool/device_d21_qdr0.02_cdr0.02/devices"
sample_dir = "device_pool/device_d21_qdr0.02_cdr0.02/samples_over_holding_options_p0.002"

min_ler_pos = {
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

def get_min_ler_pos(device:Device, samples: List[sinter.TaskStats], filter_func: Callable[['sinter.TaskStats'], Any] = lambda _: True, devide_weight = True):
    # Record the position of the minimum LER, according to specified_holding_cycle or holding_cycle_ratio
    result = Analyzer.analyze_device(device, skip_shortest_path=True)
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
    "LOCALAVG": {
        "0": calculate_cdf(min_ler_pos["LOCALAVG"]["0"]),
        "+": calculate_cdf(min_ler_pos["LOCALAVG"]["+"]),
    },
    "LOCALMAX": {
        "0": calculate_cdf(min_ler_pos["LOCALMAX"]["0"]),
        "+": calculate_cdf(min_ler_pos["LOCALMAX"]["+"]),
    },
}

# Plot SPECIFIED
ax.step(min_ler_pos_cdf["SPECIFIED"]["0"][0], min_ler_pos_cdf["SPECIFIED"]["0"][1], label="G, $|0\\rangle$", linewidth=1, color="C0")
ax.step(min_ler_pos_cdf["SPECIFIED"]["+"][0], min_ler_pos_cdf["SPECIFIED"]["+"][1], label="G, $|+\\rangle$", linewidth=1, color="C2")
# Plot vertical line at median
ax.axvline(np.median(min_ler_pos["SPECIFIED"]["0"]), color="C0", linestyle="--", label=f"Med.: {np.median(min_ler_pos['SPECIFIED']['0']):.2f}")
ax.axvline(np.median(min_ler_pos["SPECIFIED"]["+"]), color="C2", linestyle="--", label=f"Med.: {np.median(min_ler_pos['SPECIFIED']['+']):.2f}")

ax.legend(loc='lower right', fontsize=legend_fontsize)
ax.grid()
ax.set_xlabel("Sweet Point-Average Weight Ratio")
ax.set_ylabel("CDF")
ax.set_ylim(0, 1)
ax.text(*notation_positions, '(d)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

plt.tight_layout()
plt.savefig('global_vs_local.pdf', format='pdf')