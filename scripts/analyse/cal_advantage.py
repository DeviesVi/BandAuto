import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import itertools
import sinter
import pickle

dr = 0.02

def calculate_ler(sample: sinter.TaskStats):
    return sample.errors/(sample.shots - sample.discards)

plt.figure(figsize=(12, 6), dpi=100)

marker = itertools.cycle(("o", "o", "^", "^", "s", "s"))
color = itertools.cycle(
    ("#ff7f0e", "#ff7f0e", "#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c")
)

notation_positions = (-0.22, 1)
notation_fontsize = 12
markersize = 4
legend_fontsize = 8

distances = [15, 21, 27]
defect_rates = [0.005, 0.01, 0.015, 0.02]


with open("sp_data/cmp_statistics.json", "r") as f:
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
plt.xlabel("Defect Rate")
plt.ylabel("Avg. X Distance")
plt.legend(fontsize=legend_fontsize)
plt.grid()

print("X")
print("Tradition")
xt = np.average(
    [
        result["Tradition"]["x_distance"]
        for result in results[str(27)][str(dr)].values()
    ]
)
print(xt)

print("Bandage")
xb = np.average(
    [result["Bandage"]["x_distance"] for result in results[str(27)][str(dr)].values()]
)
print(xb)

print("Improvement")
print((xb - xt) / xt)

ax.text(*notation_positions, '(c)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

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
plt.xlabel("Defect Rate")
plt.ylabel("Avg. Z Distance")
plt.legend(fontsize=legend_fontsize)
plt.grid()

print("Z")
print("Tradition")
zt = np.average(
    [
        result["Tradition"]["z_distance"]
        for result in results[str(27)][str(dr)].values()
    ]
)
print(zt)

print("Bandage")
zb = np.average(
    [result["Bandage"]["z_distance"] for result in results[str(27)][str(dr)].values()]
)
print(zb)

print("Improvement")
print((zb - zt) / zt)

print("Avg improvement")
print(((xb - xt) / xt + (zb - zt) / zt) / 2)

ax.text(*notation_positions, '(d)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

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
plt.xlabel("Defect Rate")
plt.ylabel("Avg. Disabled Qubit Pct.")
plt.legend(fontsize=legend_fontsize)
plt.grid()

print('pct. disabled qubits')
print('Tradition')
print(
    np.average(
        [
            result["Tradition"]["disabled_qubit_percentage"]
            for result in results[str(27)][str(dr)].values()
        ]
    )
)
print('Bandage')
print(
    np.average(
        [
            result["Bandage"]["disabled_qubit_percentage"]
            for result in results[str(27)][str(dr)].values()
        ]
    )
)

ax.text(*notation_positions, '(e)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

ax = plt.subplot(236)

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
plt.xlabel("Defect Rate")
plt.ylabel("Avg. of Avg. Super-Stab. Weight")
plt.legend(fontsize=legend_fontsize)
plt.grid()

ax.text(*notation_positions, '(f)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)


ax = plt.subplot(231)
ax.axis('off')
ax.text(*notation_positions, '(a)', transform=ax.transAxes, va='top', fontsize=12)

ax = plt.subplot(234)

samples1: List[sinter.TaskStats] = pickle.load(open('sp_data/bcmp1.pkl', 'rb'))
samples2: List[sinter.TaskStats] = pickle.load(open('sp_data/bcmp.pkl', 'rb'))
samples3: List[sinter.TaskStats] = pickle.load(open('sp_data/bcmp3.pkl', 'rb'))

legend = {
        0: 'Bandage',
        1: 'Tradition',
    }

results = {
    '0': {
        0: [],
        1: [],
    },
    '+': {
        0: [],
        1: [],
    },
}

# Print lowest point y value for each curve
for i in range(2):
    for s in ['0', '+']:
        filter_func = lambda stat: stat.json_metadata["initial_state"] == s and stat.json_metadata["device_index"] == i
        results[s][i].append(min([calculate_ler(stat) for stat in filter(filter_func, samples1)]))


for i in range(2):
    for s in ['0', '+']:
        filter_func = lambda stat: stat.json_metadata["initial_state"] == s and stat.json_metadata["device_index"] == i
        results[s][i].append(min([calculate_ler(stat) for stat in filter(filter_func, samples2)]))

for i in range(2):
    for s in ['0', '+']:
        filter_func = lambda stat: stat.json_metadata["initial_state"] == s and stat.json_metadata["device_index"] == i
        results[s][i].append(min([calculate_ler(stat) for stat in filter(filter_func, samples3)]))

for s in ['0', '+']:
    if s == '0':
        plt.plot(range(1, 4), results[s][0], label=f'Bandage |{s}>', marker='o', color='C0')
        plt.plot(range(1, 4), results[s][1], label=f'Tradition |{s}>', marker='o', linestyle='--', color='C0')
    else:
        plt.plot(range(1, 4), results[s][0], label=f'Bandage |{s}>', marker='^', color='C2')
        plt.plot(range(1, 4), results[s][1], label=f'Tradition |{s}>', marker='^', linestyle='--', color='C2')


plt.xticks(range(1, 4), ["A", "AB", "ABC"])
plt.xlabel("Defective Data Qubits")
plt.ylabel("LER")
plt.legend(fontsize=legend_fontsize)
plt.grid()

ax.text(*notation_positions, '(b)', transform=ax.transAxes, va='top', fontsize=notation_fontsize)

plt.tight_layout()

plt.show()
