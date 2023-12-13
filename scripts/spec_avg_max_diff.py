import matplotlib.pyplot as plt
import os
from defective_surface_code_adapter import Device, Analyzer
import pickle
import sinter
from typing import List
import numpy as np

device_path = "device_pool/devices/"

min_ler_s_list = []
min_ler_m_list = []
min_ler_a_list = []

for file in os.listdir(device_path):
    device = Device.load(f"{device_path}{file}")
    samples_path = f"device_pool/samples/samples_{device.strong_id}.pkl"
    samples: List[sinter.TaskStats] = pickle.load(open(samples_path, "rb"))

    # Classify samples according to holding cycle option
    max_samples: List[sinter.TaskStats] = []
    avg_samples: List[sinter.TaskStats] = []
    spec_samples: List[sinter.TaskStats] = []
    for sample in samples:
        if sample.json_metadata["holding_cycle_option"] == "MAX":
            max_samples.append(sample)
        elif sample.json_metadata["holding_cycle_option"] == "AVG":
            avg_samples.append(sample)
        elif sample.json_metadata["holding_cycle_option"] == "SPEC":
            spec_samples.append(sample)

    # Calculate min logical error rate for each holding cycle option
    ler_m = [
        sample.errors / (sample.shots - sample.discards) for sample in max_samples
    ]
    ler_a = [
        sample.errors / (sample.shots - sample.discards) for sample in avg_samples
    ]
    ler_s = [
        sample.errors / (sample.shots - sample.discards) for sample in spec_samples
    ]
    min_ler_m = min(ler_m)
    min_ler_a = min(ler_a)
    min_ler_s = min(ler_s)

    min_ler_s_list.append(min_ler_s)
    min_ler_m_list.append(min_ler_m)
    min_ler_a_list.append(min_ler_a)

def calculate_cdf(data):
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    return data_sorted, cdf

min_ler_diff_sm = [min_ler_s - min_ler_m for min_ler_s, min_ler_m in zip(min_ler_s_list, min_ler_m_list)]
min_ler_diff_sa = [min_ler_s - min_ler_a for min_ler_s, min_ler_a in zip(min_ler_s_list, min_ler_a_list)]

min_ler_relative_diff_sm = [min_ler_diff / min_ler_m for min_ler_diff, min_ler_m in zip(min_ler_diff_sm, min_ler_m_list)]
min_ler_relative_diff_sa = [min_ler_diff / min_ler_a for min_ler_diff, min_ler_a in zip(min_ler_diff_sa, min_ler_a_list)]

min_ler_diff_sm_sorted, cdf_sm = calculate_cdf(min_ler_diff_sm)
min_ler_diff_sa_sorted, cdf_sa = calculate_cdf(min_ler_diff_sa)

min_ler_relative_diff_sm_sorted, cdf_sm = calculate_cdf(min_ler_relative_diff_sm)
min_ler_relative_diff_sa_sorted, cdf_sa = calculate_cdf(min_ler_relative_diff_sa)


plt.subplot(121)
plt.plot(min_ler_diff_sm_sorted, cdf_sm, label="SPEC-MAX", linewidth=2, color="blue")
plt.plot(min_ler_diff_sa_sorted, cdf_sa, label="SPEC-AVG", linewidth=2, color="orange")
# Plot vertical line at CDF = 0.5
plt.axvline(x=min_ler_diff_sm_sorted[int(len(min_ler_diff_sm_sorted)/2)], color="blue", linestyle="dashed")
plt.axvline(x=min_ler_diff_sa_sorted[int(len(min_ler_diff_sa_sorted)/2)], color="orange", linestyle="dashed")
plt.legend()
plt.xlabel("Min LER difference")
plt.ylabel("CDF")
plt.ylim(0, 1)

plt.subplot(122)
plt.plot(min_ler_relative_diff_sm_sorted, cdf_sm, label="(SPEC-MAX)/MAX", linewidth=2, color="blue")
plt.plot(min_ler_relative_diff_sa_sorted, cdf_sa, label="(SPEC-AVG)/AVG", linewidth=2, color="orange")
# Plot vertical line at CDF = 0.5
plt.axvline(x=min_ler_relative_diff_sm_sorted[int(len(min_ler_relative_diff_sm_sorted)/2)], color="blue", linestyle="dashed", label=f"Median={min_ler_relative_diff_sm_sorted[int(len(min_ler_relative_diff_sm_sorted)/2)]:.2f}")
plt.axvline(x=min_ler_relative_diff_sa_sorted[int(len(min_ler_relative_diff_sa_sorted)/2)], color="orange", linestyle="dashed", label=f"Median={min_ler_relative_diff_sa_sorted[int(len(min_ler_relative_diff_sa_sorted)/2)]:.2f}")

plt.legend()
plt.xlabel("Min LER relative difference")
plt.ylabel("CDF")
plt.ylim(0, 1)

plt.show()
