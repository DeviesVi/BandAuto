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


distances = [15, 21, 27]
defect_rates = [0.01, 0.02, 0.03, 0.04]


with open("data/statistics/bandage_vs_tradition_coupler_only.json", "r") as f:
    results: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = json.load(f)

print("X Distance")
print("-" * 40)
for d in distances:
    print("L =", d)
    print("Tradition")
    tradition_data = [
        np.average(
            [
                result["Tradition"]["x_distance"]
                for result in results[str(d)][str(defect_rate)].values()
            ]
        )
        for defect_rate in defect_rates
    ]
    print(tradition_data)
    print("Bandage")
    bandage_data = [
        np.average(
            [
                result["Bandage"]["x_distance"]
                for result in results[str(d)][str(defect_rate)].values()
            ]
        )
        for defect_rate in defect_rates
    ]
    print(bandage_data)
    print("-" * 40)
    print(f"Advantage of X Distance on L={d}")
    print(
        (np.array(bandage_data) - np.array(tradition_data)) / np.array(tradition_data)
    )
    print("-" * 40)


print("Z Distance")
print("-" * 40)
for d in distances:
    print("L =", d)
    print("Tradition")
    tradition_data = [
        np.average(
            [
                result["Tradition"]["z_distance"]
                for result in results[str(d)][str(defect_rate)].values()
            ]
        )
        for defect_rate in defect_rates
    ]
    print(tradition_data)
    print("Bandage")
    bandage_data = [
        np.average(
            [
                result["Bandage"]["z_distance"]
                for result in results[str(d)][str(defect_rate)].values()
            ]
        )
        for defect_rate in defect_rates
    ]
    print(bandage_data)
    print("-" * 40)
    print(f"Advantage of Z Distance on L={d}")
    print(
        (np.array(bandage_data) - np.array(tradition_data)) / np.array(tradition_data)
    )
    print("-" * 40)

print("Disabled Percentage")
print("-" * 40)
for d in distances:
    print("L =", d)
    print("Tradition")
    tradition_data = [
        np.average(
            [
                result["Tradition"]["disabled_qubit_percentage"]
                for result in results[str(d)][str(defect_rate)].values()
            ]
        )
        for defect_rate in defect_rates
    ]
    print(tradition_data)
    print("Bandage")
    bandage_data = [
        np.average(
            [
                result["Bandage"]["disabled_qubit_percentage"]
                for result in results[str(d)][str(defect_rate)].values()
            ]
        )
        for defect_rate in defect_rates
    ]
    print(bandage_data)
    print("-" * 40)
    print(f"Advantage of Disabled Percentage on L={d}")
    print(
        (np.array(bandage_data) - np.array(tradition_data)) / np.array(tradition_data)
    )
    print("-" * 40)

print("Avg Weight")
print("-" * 40)
for d in distances:
    print("L =", d)
    print("Tradition")
    tradition_data = [
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
        ]
    print(tradition_data)
    print("Bandage")
    bandage_data = [
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
        ]
    print(bandage_data)
    print("-" * 40)
    print(f"Advantage of Avg Weight on L={d}")
    print(
        (np.array(bandage_data) - np.array(tradition_data)) / np.array(tradition_data)
    )
    print("-" * 40)