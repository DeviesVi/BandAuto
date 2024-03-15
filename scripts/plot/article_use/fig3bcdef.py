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


plt.figure(figsize=(12, 6), dpi=100)

marker = itertools.cycle(("o", "o", "^", "^", "s", "s"))
color = itertools.cycle(
    ("#ff7f0e", "#ff7f0e", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c")
)

notation_positions = (-0.3, 1)
notation_fontsize = 12
markersize = 4
legend_fontsize = 8
xylabel_fontsize = 12

distances = [15, 21, 27]
defect_rates = [0.005, 0.01, 0.015, 0.02]


with open("data/statistics/bandage_vs_tradition.json", "r") as f:
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = json.load(f)

ax = plt.subplot(232)

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
    "(c)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)

ax = plt.subplot(235)

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
    "(d)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)

ax = plt.subplot(233)

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
    "(e)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)

ax = plt.subplot(236)

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
    "(f)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)


ax = plt.subplot(231)
ax.axis("off")
ax.text(*notation_positions, "(a)", transform=ax.transAxes, va="top", fontsize=12)

ax = plt.subplot(234)

samples: List[sinter.TaskStats] = pickle.load(
    open("data/samples/abc_defect/abc_defect.pkl", "rb")
)

legend = {
    0: "Bandage",
    1: "Tradition",
}

results = {
    "0": {
        False: [],
        True: [],
    },
    "+": {
        False: [],
        True: [],
    },
}

# Print lowest point y value for each curve
for i in range(3):
    for s in ["0", "+"]:
        for traditional in [False, True]:
            filter_func = (
                lambda stat: stat.json_metadata["initial_state"] == s
                and stat.json_metadata["device_index"] == i
                and stat.json_metadata["use_traditional_adapter"] == traditional
            )
            results[s][traditional].append(
                min([calculate_ler(stat) for stat in filter(filter_func, samples)])
            )

for s in ["0", "+"]:
    if s == "0":
        plt.plot(
            range(1, 4),
            results[s][False],
            label=f"B |{s}>",
            marker="o",
            color="C0",
        )
        plt.plot(
            range(1, 4),
            results[s][True],
            label=f"T |{s}>",
            marker="o",
            linestyle="--",
            color="C0",
        )
    else:
        plt.plot(
            range(1, 4),
            results[s][False],
            label=f"B |{s}>",
            marker="^",
            color="C2",
        )
        plt.plot(
            range(1, 4),
            results[s][True],
            label=f"T |{s}>",
            marker="^",
            linestyle="--",
            color="C2",
        )


plt.xticks(range(1, 4), ["A", "AB", "ABC"])
plt.xlabel("Defective Data Qubits", fontsize=xylabel_fontsize)
plt.ylabel("LER", fontsize=xylabel_fontsize)
plt.legend(fontsize=legend_fontsize)
plt.grid()

ax.text(
    *notation_positions,
    "(b)",
    transform=ax.transAxes,
    va="top",
    fontsize=notation_fontsize,
)

plt.tight_layout()

plt.savefig("fig3bcdef.pdf", format="pdf")
