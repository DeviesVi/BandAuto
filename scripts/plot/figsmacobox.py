import json
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_fontsize = 8

notation_positions = (-0.25, 1)
notation_fontsize = 12
distances = [15, 21, 27]
defect_rates = [0.01, 0.02, 0.03, 0.04]
linewidth = 0.3
flier_size = 4


with open("manuscript_data/sample_data/statistics/bandage_vs_tradition_coupler_only.json", "r") as f:
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

plt.figure(figsize=(12, 6), dpi=100)

# x_distance
ax = plt.subplot(231)
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

plt.xticks(np.arange(1, 5), defect_rates)

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("X Distance")

ax.text(*notation_positions, '(a)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# z_distance
ax = plt.subplot(234)
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

plt.xticks(np.arange(1, 5), defect_rates)

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("Z Distance")

ax.text(*notation_positions, '(b)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# disabled qubit percentage
ax = plt.subplot(232)
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

plt.xticks(np.arange(1, 5), defect_rates)

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("Disabled Qubit Pct.")

ax.text(*notation_positions, '(c)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# average super-stabilizer weight
ax = plt.subplot(235)
for d in distances:
    plt.boxplot(
        [
            [
                result["Bandage"]["stabilizer_statistics"]["avg_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Bandage"]["stabilizer_statistics"]["avg_weight"] is not None
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
                result["Tradition"]["stabilizer_statistics"]["avg_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Tradition"]["stabilizer_statistics"]["avg_weight"] is not None
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

plt.xticks(np.arange(1, 5), defect_rates)

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("Avg. Super-Stab. Weight")

ax.text(*notation_positions, '(d)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

# average super-stabilizer weight no flier
ax = plt.subplot(233)
for d in distances:
    plt.boxplot(
        [
            [
                result["Bandage"]["stabilizer_statistics"]["avg_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Bandage"]["stabilizer_statistics"]["avg_weight"] is not None
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
                result["Tradition"]["stabilizer_statistics"]["avg_weight"]
                for result in results[str(d)][str(dr)].values()
                if result["Tradition"]["stabilizer_statistics"]["avg_weight"] is not None
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

plt.xticks(np.arange(1, 5), defect_rates)

legend_handles = [
    Patch(facecolor=color_box[d], edgecolor='black', label=f"B, L={d}", linewidth=linewidth) for d in distances
] + [
    Patch(facecolor=color_box_dark[d], edgecolor='black', label=f"T, L={d}", linewidth=linewidth) for d in distances
]

plt.legend(handles=legend_handles, ncol=2, fontsize=legend_fontsize)

plt.grid()
plt.xlabel("Defect Rate")
plt.ylabel("Avg. Super-Stab. Weight")

ax.text(*notation_positions, '(e)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

plt.tight_layout()

plt.savefig('figsmacobox.pdf', format='pdf')