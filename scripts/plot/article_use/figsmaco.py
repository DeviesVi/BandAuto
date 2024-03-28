import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import itertools
import sinter
import pickle


def calculate_ler(sample: sinter.TaskStats):
    return sample.errors / (sample.shots - sample.discards)


plt.figure(figsize=(8, 6), dpi=100)

marker = itertools.cycle(("o", "o", "^", "^", "s", "s"))
color = itertools.cycle(
    ("#ff7f0e", "#ff7f0e", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c")
)

notation_positions = (-0.3, 1)
notation_fontsize = 16
markersize = 4
legend_fontsize = 8
xylabel_fontsize = 12

distances = [15, 21, 27]
defect_rates = [0.01, 0.02, 0.03, 0.04]


with open("data/statistics/bandage_vs_tradition_coupler_only.json", "r") as f:
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = json.load(f)

ax = plt.subplot(221)

for d in distances:
    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Tradition"]["x_distance"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"T, L={d}",
        marker=next(marker),
        markersize=markersize,
        color=next(color),
        linestyle="--",
    )

    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Bandage"]["x_distance"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"B, L={d}",
        marker=next(marker),
        markersize=markersize,
        color=next(color),
    )

plt.xticks(defect_rates)
plt.xlabel("Defect Rate", fontsize=xylabel_fontsize)
plt.ylabel("Avg. X Distance", fontsize=xylabel_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()

ax.text(
    *notation_positions,
    "(a)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)

ax = plt.subplot(223)

for d in distances:
    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Tradition"]["z_distance"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"T, L={d}",
        marker=next(marker),
        markersize=markersize,
        color=next(color),
        linestyle="--",
    )

    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Bandage"]["z_distance"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"B, L={d}",
        marker=next(marker),
        markersize=markersize,
        color=next(color),
    )

plt.xticks(defect_rates)
plt.xlabel("Defect Rate", fontsize=xylabel_fontsize)
plt.ylabel("Avg. Z Distance", fontsize=xylabel_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()

ax.text(
    *notation_positions,
    "(b)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)

ax = plt.subplot(222)

for d in distances:
    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Tradition"]["disabled_qubit_percentage"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"T, L={d}",
        marker=next(marker),
        markersize=markersize,
        color=next(color),
        linestyle="--",
    )

    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Bandage"]["disabled_qubit_percentage"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"B, L={d}",
        marker=next(marker),
        markersize=markersize,
        color=next(color),
    )

plt.xticks(defect_rates)
plt.xlabel("Defect Rate", fontsize=xylabel_fontsize)
plt.ylabel("Avg. Disabled Qubit Pct.", fontsize=xylabel_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()

ax.text(
    *notation_positions,
    "(c)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)

ax = plt.subplot(224)

for d in distances:
    plt.plot(
        defect_rates,
        [
            np.sum(
                [
                    result["Tradition"]["stabilizer_statistics"]["total_weight"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            / np.sum(
                [
                    result["Tradition"]["stabilizer_statistics"]["total_count"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"T, L={d}",
        marker=next(marker),
        markersize=markersize,
        color=next(color),
        linestyle="--",
    )

    plt.plot(
        defect_rates,
        [
            np.sum(
                [
                    result["Bandage"]["stabilizer_statistics"]["total_weight"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            / np.sum(
                [
                    result["Bandage"]["stabilizer_statistics"]["total_count"]
                    for result in results[str(d)][str(defect_rate)].values()
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"B, L={d}",
        marker=next(marker),
        markersize=markersize,
        color=next(color),
    )

plt.xticks(defect_rates)
plt.xlabel("Defect Rate", fontsize=xylabel_fontsize)
plt.ylabel("Avg. Super-Stab. Weight", fontsize=xylabel_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()

ax.text(
    *notation_positions,
    "(d)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)

plt.tight_layout()

plt.savefig("figsmaco.pdf", format="pdf")
