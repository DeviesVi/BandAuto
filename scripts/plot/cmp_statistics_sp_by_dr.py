import json
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_fontsize = 8


distances = [15, 21, 27]
defect_rates = [0.005, 0.01, 0.015, 0.02]
linewidth = 0.3
flier_size = 4


with open("sp_data/cmp_statistics.json", "r") as f:
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = json.load(f)


offset = {
    15: -0.18,
    21: -0.12,
    27: -0.06,
}

box_width = 0.04

color_box = {
    15: '#ffbb78',
    21: '#aec7e8',
    27: '#98df8a',
}

color_box_dark = {
    15: '#d95f02',
    21: '#1f77b4',
    27: '#2ca02c',
}

# x_distance
plt.figure(figsize=(8, 6), dpi=100)
for d in distances:
    plt.boxplot(
        [
            [
                result["Bandage"]["x_distance"]
                for result in results[str(d)][str(dr)].values()
            ]
            for dr in defect_rates
        ],
        positions=[i + 1 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["x_distance"]
                for result in results[str(d)][str(dr)].values()
            ]
            for dr in defect_rates
        ],
        positions=[i + 1.3 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

plt.xticks(np.arange(1, 5), [0.005, 0.01, 0.015, 0.02])

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("X Distance")
plt.savefig('x_distance_sp.pdf', format='pdf')

# z_distance
plt.figure(figsize=(8, 6), dpi=100)
for d in distances:
    plt.boxplot(
        [
            [
                result["Bandage"]["z_distance"]
                for result in results[str(d)][str(dr)].values()
            ]
            for dr in defect_rates
        ],
        positions=[i + 1 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["z_distance"]
                for result in results[str(d)][str(dr)].values()
            ]
            for dr in defect_rates
        ],
        positions=[i + 1.3 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

plt.xticks(np.arange(1, 5), [0.005, 0.01, 0.015, 0.02])

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("Z Distance")
plt.savefig('z_distance_sp.pdf', format='pdf')

# disabled qubit percentage
plt.figure(figsize=(8, 6), dpi=100)
for d in distances:
    plt.boxplot(
        [
            [
                result["Bandage"]["disabled_qubit_percentage"]
                for result in results[str(d)][str(dr)].values()
            ]
            for dr in defect_rates
        ],
        positions=[i + 1 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["disabled_qubit_percentage"]
                for result in results[str(d)][str(dr)].values()
            ]
            for dr in defect_rates
        ],
        positions=[i + 1.3 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

plt.xticks(np.arange(1, 5), [0.005, 0.01, 0.015, 0.02])

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("Disabled Qubit Pct.")
plt.savefig('percentage_sp.pdf', format='pdf')

# average super-stabilizer weight
plt.figure(figsize=(8, 6), dpi=100)
for d in distances:
    plt.boxplot(
        [
            [
                result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
            ]
            for dr in defect_rates
        ],
        positions=[i + 1 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Tradition"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
            ]
            for dr in defect_rates
        ],
        positions=[i + 1.3 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
    )

plt.xticks(np.arange(1, 5), [0.005, 0.01, 0.015, 0.02])

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("Avg. Super-Stab. Weight")
plt.savefig('weight_sp.pdf', format='pdf')

# average super-stabilizer weight no flier
plt.figure(figsize=(8, 6), dpi=100)
for d in distances:
    plt.boxplot(
        [
            [
                result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
            ]
            for dr in defect_rates
        ],
        positions=[i + 1 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
        showfliers=False,
    )

    plt.boxplot(
        [
            [
                result["Tradition"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Tradition"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
            ]
            for dr in defect_rates
        ],
        positions=[i + 1.3 + offset[d] for i, dr in enumerate(defect_rates)],
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor=color_box_dark[d], edgecolor="black", linewidth=linewidth),
        medianprops=dict(color="black"),
        flierprops=dict(marker=".", markerfacecolor=color_box[d], markeredgewidth=linewidth, markersize=flier_size),
        whiskerprops=dict(linewidth=linewidth),
        capprops=dict(linewidth=linewidth),
        showfliers=False,
    )

plt.xticks(np.arange(1, 5), [0.005, 0.01, 0.015, 0.02])

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("Avg. Super-Stab. Weight")
plt.savefig('weight_sp_no_flier.pdf', format='pdf')