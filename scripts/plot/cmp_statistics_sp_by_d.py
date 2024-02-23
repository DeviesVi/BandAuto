import json
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_fontsize = 8


distances = [15, 21, 27]
defect_rates = [0.005, 0.01, 0.015, 0.02]


with open("sp_data/cmp_statistics.json", "r") as f:
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = json.load(f)


offset = {
    0.005: -0.06,
    0.01: -0.12,
    0.015: -0.18,
    0.02: -0.24,
}

box_width = 0.04

color_box = {
    0.02: '#ff9896',
    0.015: '#ffbb78',
    0.01: '#aec7e8',
    0.005: '#98df8a',
}

color_box_dark = {
    0.02: '#d62728',
    0.015: '#d95f02',
    0.01: '#1f77b4',
    0.005: '#2ca02c',
}

# x_distance
plt.figure(figsize=(8, 6), dpi=100)
for dr in defect_rates:
    plt.boxplot(
        [
            [
                result["Bandage"]["x_distance"]
                for result in results[str(d)][str(dr)].values()
            ]
            for d in distances
        ],
        positions=[i + 1 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["x_distance"]
                for result in results[str(d)][str(dr)].values()
            ]
            for d in distances
        ],
        positions=[i + 1.3 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
    )

plt.xticks(np.arange(1, 4), [15, 21, 27])

legend_handles = [
    Patch(facecolor=color_box[dr], edgecolor='black', label=f"B, DR={dr}") for dr in defect_rates
] + [
    Patch(facecolor=color_box_dark[dr], edgecolor='black', label=f"T, DR={dr}") for dr in defect_rates
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Code Size $L$")
plt.ylabel("X Distance")
plt.savefig('x_distance_sp.pdf', format='pdf')


# z_distance
plt.figure(figsize=(8, 6), dpi=100)
for dr in defect_rates:
    plt.boxplot(
        [
            [
                result["Bandage"]["z_distance"]
                for result in results[str(d)][str(dr)].values()
            ]
            for d in distances
        ],
        positions=[i + 1 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["z_distance"]
                for result in results[str(d)][str(dr)].values()
            ]
            for d in distances
        ],
        positions=[i + 1.3 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
    )

plt.xticks(np.arange(1, 4), [15, 21, 27])

legend_handles = [
    Patch(facecolor=color_box[dr], edgecolor='black', label=f"B, DR={dr}") for dr in defect_rates
] + [
    Patch(facecolor=color_box_dark[dr], edgecolor='black', label=f"T, DR={dr}") for dr in defect_rates
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Code Size $L$")
plt.ylabel("Z Distance")
plt.savefig('z_distance_sp.pdf', format='pdf')

# disabled qubit percentage
plt.figure(figsize=(8, 6), dpi=100)
for dr in defect_rates:
    plt.boxplot(
        [
            [
                result["Bandage"]["disabled_qubit_percentage"]
                for result in results[str(d)][str(dr)].values()
            ]
            for d in distances
        ],
        positions=[i + 1 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["disabled_qubit_percentage"]
                for result in results[str(d)][str(dr)].values()
            ]
            for d in distances
        ],
        positions=[i + 1.3 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
    )

plt.xticks(np.arange(1, 4), [15, 21, 27])

legend_handles = [
    Patch(facecolor=color_box[dr], edgecolor='black', label=f"B, DR={dr}") for dr in defect_rates
] + [
    Patch(facecolor=color_box_dark[dr], edgecolor='black', label=f"T, DR={dr}") for dr in defect_rates
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Code Size $L$")
plt.ylabel("Disabled Qubit Pct.")
plt.savefig('percentage_sp.pdf', format='pdf')

# avg super-stabilizer weight
plt.figure(figsize=(8, 6), dpi=100)
for dr in defect_rates:
    plt.boxplot(
        [
            [
                result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
            ]
            for d in distances
        ],
        positions=[i + 1 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
            ]
            for d in distances
        ],
        positions=[i + 1.3 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
    )

plt.xticks(np.arange(1, 4), [15, 21, 27])

legend_handles = [
    Patch(facecolor=color_box[dr], edgecolor='black', label=f"B, DR={dr}") for dr in defect_rates
] + [
    Patch(facecolor=color_box_dark[dr], edgecolor='black', label=f"T, DR={dr}") for dr in defect_rates
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Code Size $L$")
plt.ylabel("Avg. Super-Stab. Weight")
plt.savefig('weight_sp.pdf', format='pdf')

# avg super-stabilizer weight no flier
plt.figure(figsize=(8, 6), dpi=100)
for dr in defect_rates:
    plt.boxplot(
        [
            [
                result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
            ]
            for d in distances
        ],
        positions=[i + 1 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
        showfliers=False,
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
            ]
            for d in distances
        ],
        positions=[i + 1.3 + offset[dr] for i, d in enumerate(distances)],
        widths=box_width,
        labels=[15, 21, 27],
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[dr], edgecolor="black"),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[dr]),
        showfliers=False,
    )

plt.xticks(np.arange(1, 4), [15, 21, 27])

legend_handles = [
    Patch(facecolor=color_box[dr], edgecolor='black', label=f"B, DR={dr}") for dr in defect_rates
] + [
    Patch(facecolor=color_box_dark[dr], edgecolor='black', label=f"T, DR={dr}") for dr in defect_rates
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Code Size $L$")
plt.ylabel("Avg. Super-Stab. Weight")
plt.savefig('weight_sp_no_flier.pdf', format='pdf')