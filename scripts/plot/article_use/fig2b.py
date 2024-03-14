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
    return sample.errors / (sample.shots - sample.discards)


notation_positions = (-0.25, 1)
notation_fontsize = 12
legend_fontsize = 8
marker_size = 4

plt.figure(figsize=(4, 2.5), dpi=100)

ax = plt.subplot(111)

distance = [15, 21, 27]
defect_rates = [0.02, 0.015, 0.01, 0.005]

device_paths = {
    (d, dr): f"device_pool/device_d{d}_qdr{dr}_cdr{dr}/devices"
    for d in distance
    for dr in defect_rates
}

sample_paths = {
    (d, dr): (
        f"device_pool/device_d{d}_qdr{dr}_cdr{dr}/samples_over_specified_cycle"
        if not (d == 21 and dr == 0.02)
        else f"device_pool/device_d{d}_qdr{dr}_cdr{dr}/samples_over_holding_options_p0.002"
    )
    for d in distance
    for dr in defect_rates
}

perfect_device_paths = [
    "device_pool/device_d15_qdr0_cdr0/devices/device_04c4a53224db00ca8c1b5ff40db07fe792268764a5f84d737d09923fb9eb8c2e.pkl",
    "device_pool/device_d21_qdr0_cdr0/devices/device_0466bb5cca000f36704b3eee72ea19060f40c8008a0dda96661f3f30c54b39ba.pkl",
    "device_pool/device_d27_qdr0_cdr0/devices/device_3eb43fcb57726a6e82badc1a68104382c1e03971ab65a13f1fd3849cca422ff0.pkl",
]

perfect_SI1000_p = [0.002, 0.003, 0.004]

perfect_sample_paths = (
    "device_pool/device_d{d}_qdr0_cdr0/samples_p{p}/samples_{strong_id}.pkl"
)

results = {
    (d, dr): []
    for d in distance
    for dr in defect_rates
}

for (d, dr), device_path in device_paths.items():
    for device_file in os.listdir(device_path):
        device = Device.load(f"{device_path}/{device_file}")
        samples: List[sinter.TaskStats] = pickle.load(
            open(f"{sample_paths[(d, dr)]}/samples_{device.strong_id}.pkl", "rb")
        )
        # Calculate min LER for each device
        min_ler_zero = 1
        min_ler_plus = 1
        for sample in samples:
            if sample.json_metadata["holding_cycle_option"] == "SPECIFIED":
                if sample.json_metadata["initial_state"] == "0":
                    min_ler_zero = min(min_ler_zero, calculate_ler(sample))
                elif sample.json_metadata["initial_state"] == "+":
                    min_ler_plus = min(min_ler_plus, calculate_ler(sample))
                else:
                    raise ValueError("Invalid initial state")
        min_ler = (min_ler_zero + min_ler_plus)/2
        results[(d, dr)].append(min_ler)

perfect_results = {(d, p): [] for p in perfect_SI1000_p for d in distance}

for i, device_path in enumerate(perfect_device_paths):
    device = Device.load(device_path)
    d = device.data_width
    for p in perfect_SI1000_p:
        samples: List[sinter.TaskStats] = pickle.load(
            open(
                perfect_sample_paths.format(d=d, p=p, strong_id=device.strong_id), "rb"
            )
        )
        # Calculate min LER for each device
        min_ler_zero = 1
        min_ler_plus = 1
        for sample in samples:
            if sample.json_metadata["holding_cycle_option"] == "SPECIFIED":
                if sample.json_metadata["initial_state"] == "0":
                    min_ler_zero = min(min_ler_zero, calculate_ler(sample))
                elif sample.json_metadata["initial_state"] == "+":
                    min_ler_plus = min(min_ler_plus, calculate_ler(sample))
                else:
                    raise ValueError("Invalid initial state")
        min_ler = (min_ler_zero + min_ler_plus)/2
        perfect_results[device.data_width, p].append(min_ler)

# Draw perfect reference
color_line = {
    0.002: "#2ca02c",
    0.003: "#1f77b4",
    0.004: "#d62728",
}

markers = {
    0.002: "o",
    0.003: "s",
    0.004: "D",
}

for i, p in enumerate(perfect_SI1000_p):
    plt.plot(
        [1, 2, 3],
        [perfect_results[15, p], perfect_results[21, p], perfect_results[27, p]],
        label=f"Perfect SI1000 p={p}",
        color=color_line[p],
        marker=markers[p],
        linestyle="--",
        markersize=marker_size,
        linewidth=1,
    )

# Draw boxplot
offset = {
    0.005: -0.06,
    0.01: -0.12,
    0.015: -0.18,
    0.02: -0.24,
}

color_box = {
    0.02: "#ff9896",
    0.015: "#ffbb78",
    0.01: "#aec7e8",
    0.005: "#98df8a",
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
        boxprops=dict(facecolor=color_box[dr], color="black", linewidth=linewidth),
        medianprops=dict(color="black", linewidth=linewidth),
        flierprops=dict(
            marker=".",
            markerfacecolor=color_box[dr],
            markersize=ms,
            markeredgewidth=linewidth,
        ),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

plt.xticks(np.arange(1, 4), [15, 21, 27])

# set y to log scale
plt.yscale("log")
legend_handles = [
    Patch(
        facecolor=color_box[dr],
        edgecolor="black",
        label=f"DR={dr}",
        linewidth=linewidth,
    )
    for dr in defect_rates
]

legend_handles2 = [
    Line2D(
        [0],
        [0],
        label=f"p={p}",
        color=color_line[p],
        marker=markers[p],
        markersize=marker_size,
        linestyle="--",
        linewidth=1,
    )
    for p in perfect_SI1000_p
]

legend = plt.legend(handles=legend_handles, fontsize=legend_fontsize, loc="lower left")

plt.grid()
plt.xlabel("Code Size $L$")
plt.ylabel("LER")

plt.tight_layout()
plt.savefig("fig2b.pdf", format="pdf")
