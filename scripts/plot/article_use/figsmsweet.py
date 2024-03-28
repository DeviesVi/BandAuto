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

plt.figure(figsize=(4, 4), dpi=100)

# fig a
print('Making fig a...')
ax = plt.subplot(211)

device_path = 'device_pool/qubit_equal_coupler/device_d21_qdr0.02_cdr0.02/devices/device_f18efd2f3ded1a49e8011fd565ecb19bf277c3010172993707f78c633d2dac40.pkl'
global_sample_dir = 'data/samples/global_shell/samples_d21_qdr0.02_cdr0.02_p0.002'
local_sample_dir = 'data/samples/local_shell/samples_d21_qdr0.02_cdr0.02_p0.002'

device = Device.load(device_path)
global_samples_path = f"{global_sample_dir}/samples_{device.strong_id}.pkl"
local_samples_path = f"{local_sample_dir}/samples_{device.strong_id}.pkl"
global_samples: List[sinter.TaskStats] = pickle.load(open(global_samples_path, "rb"))
local_samples: List[sinter.TaskStats] = pickle.load(open(local_samples_path, "rb"))

# Classify samples according to holding cycle option and initial state
max_samples_zero: List[sinter.TaskStats] = []
max_samples_plus: List[sinter.TaskStats] = []
avg_samples_zero: List[sinter.TaskStats] = []
avg_samples_plus: List[sinter.TaskStats] = []
spec_samples_zero: List[sinter.TaskStats] = []
spec_samples_plus: List[sinter.TaskStats] = []
for sample in global_samples + local_samples:
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
ax.set_xlabel("$n_{\\text{shell}}$")
ax.legend(fontsize=legend_fontsize)
ax.text(*notation_positions, '(a)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# fig b
print('Making fig b...')
ax = plt.subplot(212)

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
ax.set_xlabel("r")
ax.legend(fontsize=legend_fontsize)
ax.text(*notation_positions, '(b)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

plt.tight_layout()
plt.savefig('figsmsweet.pdf', format='pdf')