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

plt.figure(figsize=(12, 3), dpi=100)

# fig a
print('Making fig a...')
ax = plt.subplot(131)

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

# fig a
print('Making fig b...')
ax = plt.subplot(132)

device_path = 'device_pool/device_d21_qdr0.02_cdr0.02/devices'
shell_sample_path = 'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_p'
no_shell_sample_path = 'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_p_no_shell'
ratio_sample_path = 'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_p_0.36_ratio'

# Calculate LER for different p values

shell_samples_list: List[List[sinter.TaskStats]] = []
for file in os.listdir(shell_sample_path):
    shell_samples_list.append(pickle.load(open(f'{shell_sample_path}/{file}', 'rb')))

no_shell_samples_list: List[List[sinter.TaskStats]] = []
for file in os.listdir(no_shell_sample_path):
    no_shell_samples_list.append(pickle.load(open(f'{no_shell_sample_path}/{file}', 'rb')))

ratio_samples_list: List[List[sinter.TaskStats]] = []
for file in os.listdir(ratio_sample_path):
    ratio_samples_list.append(pickle.load(open(f'{ratio_sample_path}/{file}', 'rb')))

shell_p_ler_pair: List[Tuple[float, float]] = []
no_shell_p_ler_pair: List[Tuple[float, float]] = []
ratio_p_ler_pair: List[Tuple[float, float]] = []

for shell_samples in shell_samples_list:
    for shell_sample in shell_samples:
        shell_p_ler_pair.append((shell_sample.json_metadata['physical_errors']['u2'], calculate_ler(shell_sample)))

for no_shell_sample in no_shell_samples_list:
    for no_shell_sample in no_shell_sample:
        no_shell_p_ler_pair.append((no_shell_sample.json_metadata['physical_errors']['u2'], calculate_ler(no_shell_sample)))

for ratio_samples in ratio_samples_list:
    for ratio_sample in ratio_samples:
        ratio_p_ler_pair.append((ratio_sample.json_metadata['physical_errors']['u2'], calculate_ler(ratio_sample)))

# Get all unique p values sorted
shell_p = sorted(list(set([pair[0] for pair in shell_p_ler_pair])))
no_shell_p = sorted(list(set([pair[0] for pair in no_shell_p_ler_pair])))
ratio_p = sorted(list(set([pair[0] for pair in ratio_p_ler_pair])))

# Get all LER values for each p value
shell_ler = []
no_shell_ler = []
ratio_ler = []

for p in shell_p:
    shell_ler.append([pair[1] for pair in shell_p_ler_pair if pair[0] == p])

for p in no_shell_p:
    no_shell_ler.append([pair[1] for pair in no_shell_p_ler_pair if pair[0] == p])

for p in ratio_p:
    ratio_ler.append([pair[1] for pair in ratio_p_ler_pair if pair[0] == p])

linewidth = 0.3
ms = 2

# Draw boxplot
plt.boxplot(
    shell_ler,
    positions=np.arange(len(shell_p)) - 0.1,
    widths=0.12,
    patch_artist=True,
    boxprops=dict(color='black', facecolor='#2ca02c', linewidth=linewidth),
    medianprops=dict(color='black', linewidth=linewidth),
    flierprops=dict(marker='.', markersize=ms, markerfacecolor='#2ca02c', markeredgewidth=linewidth),
    whiskerprops=dict(linewidth=linewidth),
    capprops=dict(linewidth=linewidth),
)
plt.boxplot(
    no_shell_ler,
    positions=np.arange(len(no_shell_p)) + 0.1,
    widths=0.12,
    patch_artist=True,
    boxprops=dict(color='black', facecolor='#d62728', linewidth=linewidth),
    medianprops=dict(color='black', linewidth=linewidth),
    flierprops=dict(marker='.', markersize=ms, markerfacecolor='#d62728', markeredgewidth=linewidth),
    whiskerprops=dict(linewidth=linewidth),
    capprops=dict(linewidth=linewidth),
)

plt.xticks(np.arange(len(shell_p)), np.around(shell_p, 4)*1000)
# Set y to log scale
plt.yscale('log')

plt.legend(
    handles=[
        Patch(facecolor='#2ca02c', edgecolor='black', label='Sweet Point Shell', linewidth=linewidth),
        Patch(facecolor='#d62728', edgecolor='black', label='Shell Size = 1 (No Shell)', linewidth=linewidth),
    ],
    loc='lower right',
    fontsize=legend_fontsize,
)

# Set labels
ax.set_xlabel(r'p for SI1000 EM($10^{-3}$)')
ax.set_ylabel('LER')
ax.text(*notation_positions, '(b)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# fig b
print('Making fig c...')
ax = plt.subplot(133)

distance = [15, 21, 27]
defect_rates = [0.02, 0.015, 0.01, 0.005]

device_paths = {
    (d, dr):f"device_pool/device_d{d}_qdr{dr}_cdr{dr}/devices"
    for d in distance
    for dr in defect_rates
}

sample_paths = {
    (d, dr):f"device_pool/device_d{d}_qdr{dr}_cdr{dr}/samples_over_specified_cycle"
    if not (d == 21 and dr == 0.02)
    else f"device_pool/device_d{d}_qdr{dr}_cdr{dr}/samples_over_holding_options_p0.002"
    for d in distance
    for dr in defect_rates
}

perfect_device_paths = [
    'device_pool/device_d15_qdr0_cdr0/devices/device_04c4a53224db00ca8c1b5ff40db07fe792268764a5f84d737d09923fb9eb8c2e.pkl',
    'device_pool/device_d21_qdr0_cdr0/devices/device_0466bb5cca000f36704b3eee72ea19060f40c8008a0dda96661f3f30c54b39ba.pkl',
    'device_pool/device_d27_qdr0_cdr0/devices/device_3eb43fcb57726a6e82badc1a68104382c1e03971ab65a13f1fd3849cca422ff0.pkl',
]

perfect_SI1000_p = [0.002, 0.003, 0.004]

perfect_sample_paths = 'device_pool/device_d{d}_qdr0_cdr0/samples_p{p}/samples_{strong_id}.pkl'



results = {
    (d, dr):[]
    for d in distance
    for dr in defect_rates
}

for (d, dr), device_path in device_paths.items():
    for device_file in os.listdir(device_path):
        device = Device.load(f'{device_path}/{device_file}')
        samples: List[sinter.TaskStats] = pickle.load(open(f'{sample_paths[(d, dr)]}/samples_{device.strong_id}.pkl', 'rb'))
        # Calculate min LER for each device
        min_ler = 1
        for sample in samples:
            if sample.json_metadata['holding_cycle_option'] == 'SPECIFIED':
                min_ler = min(min_ler, calculate_ler(sample))
        results[(d, dr)].append(min_ler)

perfect_results = {
    (d, p): []
    for p in perfect_SI1000_p
    for d in distance
}

for i, device_path in enumerate(perfect_device_paths):
    device = Device.load(device_path)
    d = device.data_width
    for p in perfect_SI1000_p:
        samples: List[sinter.TaskStats] = pickle.load(open(perfect_sample_paths.format(d=d, p=p, strong_id=device.strong_id), 'rb'))
        # Calculate min LER for each device
        min_ler = 1
        for sample in samples:
            if sample.json_metadata['holding_cycle_option'] == 'SPECIFIED':
                min_ler = min(min_ler, calculate_ler(sample))
        perfect_results[device.data_width, p].append(min_ler)


# Draw perfect reference
color_line = {
    0.002: '#2ca02c',
    0.003: '#1f77b4',
    0.004: '#d62728',
}

markers = {
    0.002: 'o',
    0.003: 's',
    0.004: 'D',
}

for i, p in enumerate(perfect_SI1000_p):
    plt.plot(
        [1, 2, 3],
        [perfect_results[15, p], perfect_results[21, p], perfect_results[27, p]],
        label = f'Perfect SI1000 p={p}',
        color = color_line[p],
        marker = markers[p],
        linestyle='--',
        markersize = marker_size,
        linewidth = 1,
    )

# Draw boxplot
offset = {
    0.005: -0.06,
    0.01: -0.12,
    0.015: -0.18,
    0.02: -0.24,
}

color_box = {
    0.02: '#ff9896',
    0.015: '#ffbb78',
    0.01: '#aec7e8',
    0.005: '#98df8a',
}

linewidth = 0.3
ms = 2

for dr in defect_rates:
    plt.boxplot(
        [results[(d, dr)] for d in distance],
        positions=[i + 1 + offset[dr] for i, d in enumerate(distance)],
        widths=0.04,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box[dr], color='black', linewidth=linewidth),
        medianprops=dict(color='black', linewidth=linewidth),
        flierprops=dict(marker='.', markerfacecolor=color_box[dr], markersize=ms, markeredgewidth=linewidth),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

plt.xticks(np.arange(1, 4), [15, 21, 27])

# set y to log scale
plt.yscale('log')
legend_handles = [
    Patch(facecolor=color_box[dr], edgecolor='black', label=f'DR={dr}', linewidth=linewidth)
    for dr in defect_rates
]

legend_handles2 = [Line2D([0], [0], label=f'p={p}', color=color_line[p], marker=markers[p], markersize=marker_size, linestyle='--', linewidth=1) for p in perfect_SI1000_p]

legend = plt.legend(handles=legend_handles, fontsize=legend_fontsize, loc='lower left')

plt.grid()
plt.xlabel('Code Size $L$')
plt.ylabel('LER')

ax.text(*notation_positions, '(c)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

plt.tight_layout()
plt.savefig('global_performance.pdf', format='pdf')