import json
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import itertools

marker = itertools.cycle(("*", "o", "s", "D", "X", "P"))

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
    )

plt.xticks(defect_rates)
plt.xlabel("Defect rate")
plt.ylabel("Average X Distance")
plt.legend(fontsize='8')
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
    )

plt.xticks(defect_rates)
plt.xlabel("Defect rate")
plt.ylabel("Average Z Distance")
plt.legend(fontsize='8')
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
    )

plt.xticks(defect_rates)
plt.xlabel("Defect rate")
plt.ylabel("Average Disabled Qubit Percentage")
plt.legend(fontsize='8')
plt.grid()

plt.subplot(224)

for d in distances:
    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Tradition"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                    for result in results[str(d)][str(defect_rate)].values()
                    if result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"T, L={d}",
        marker=next(marker),
        linestyle="--",
    )

    plt.plot(
        defect_rates,
        [
            np.average(
                [
                    result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"]
                    for result in results[str(d)][str(defect_rate)].values()
                    if result["Bandage"]["stabilizer_statistics"]["avg_stabilizer_weight"] is not None
                ]
            )
            for defect_rate in defect_rates
        ],
        label=f"B, L={d}",
        marker=next(marker),
    )

plt.xticks(defect_rates)
plt.xlabel("Defect rate")
plt.ylabel("Average Super-Stabilizer Weight")
plt.legend(fontsize='8')
plt.grid()

plt.show()
