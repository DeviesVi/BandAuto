import json
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import itertools

plt.figure(figsize=(12, 9), dpi=100)

marker = itertools.cycle(("o", "o", "^", "^", "s", "s"))
color = itertools.cycle(
    ("#ff7f0e", "#ff7f0e", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c")
)

markersize = 4
legend_fontsize = 10

distances = [15, 21, 27]
defect_rates = [0.005, 0.01, 0.015, 0.02]


with open("sp_data/cmp_statistics.json", "r") as f:
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = json.load(f)

plt.subplot(221)

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
plt.xlabel("Defect rate")
plt.ylabel("Average X Distance")
plt.legend(fontsize=legend_fontsize)
plt.grid()

plt.subplot(222)

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
plt.xlabel("Defect rate")
plt.ylabel("Average Z Distance")
plt.legend(fontsize=legend_fontsize)
plt.grid()

plt.subplot(223)

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
plt.xlabel("Defect rate")
plt.ylabel("Average Disabled Qubit Percentage")
plt.legend(fontsize=legend_fontsize)
plt.grid()

plt.subplot(224)

for d in distances:
    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Tradition"]["stabilizer_statistics"][
                        "avg_stabilizer_weight"
                    ]
                    for result in results[str(d)][str(defect_rate)].values()
                    if result["Bandage"]["stabilizer_statistics"][
                        "avg_stabilizer_weight"
                    ]
                    is not None
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
                    result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                    for result in results[str(d)][str(defect_rate)].values()
                    if result["Bandage"]["stabilizer_statistics"][
                        "avg_stabilizer_weight"
                    ]
                    is not None
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
plt.xlabel("Defect rate")
plt.ylabel("Average Super-Stabilizer Weight")
plt.legend(fontsize=legend_fontsize)
plt.grid()


plt.savefig("cmp_statistics.pdf", format="pdf")
