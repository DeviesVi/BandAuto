device_path = 'device_pool/device_d21_qdr0.02_cdr0.02/devices'
shell_sample_path = 'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_p'
no_shell_sample_path = 'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_p_no_shell'
ratio_sample_path = 'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_p_0.36_ratio'

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os
import sinter
import numpy as np
import pickle
from typing import List, Tuple

from utils import calculate_ler

# Calculate LER for different p values

shell_samples_list: List[List[sinter.TaskStats]] = []
for file in os.listdir(shell_sample_path):
    shell_samples_list.append(pickle.load(open(f'{shell_sample_path}/{file}', 'rb')))

no_shell_samples_list: List[List[sinter.TaskStats]] = []
for file in os.listdir(no_shell_sample_path):
    no_shell_samples_list.append(pickle.load(open(f'{no_shell_sample_path}/{file}', 'rb')))

ratio_samples_list: List[List[sinter.TaskStats]] = []
for file in os.listdir(ratio_sample_path):
    ratio_samples_list.append(pickle.load(open(f'{ratio_sample_path}/{file}', 'rb')))

shell_p_ler_pair: List[Tuple[float, float]] = []
no_shell_p_ler_pair: List[Tuple[float, float]] = []
ratio_p_ler_pair: List[Tuple[float, float]] = []

for shell_samples in shell_samples_list:
    for shell_sample in shell_samples:
        shell_p_ler_pair.append((shell_sample.json_metadata['physical_errors']['u2'], calculate_ler(shell_sample)))

for no_shell_sample in no_shell_samples_list:
    for no_shell_sample in no_shell_sample:
        no_shell_p_ler_pair.append((no_shell_sample.json_metadata['physical_errors']['u2'], calculate_ler(no_shell_sample)))

for ratio_samples in ratio_samples_list:
    for ratio_sample in ratio_samples:
        ratio_p_ler_pair.append((ratio_sample.json_metadata['physical_errors']['u2'], calculate_ler(ratio_sample)))

# Get all unique p values sorted
shell_p = sorted(list(set([pair[0] for pair in shell_p_ler_pair])))
no_shell_p = sorted(list(set([pair[0] for pair in no_shell_p_ler_pair])))
ratio_p = sorted(list(set([pair[0] for pair in ratio_p_ler_pair])))

# Get all LER values for each p value
shell_ler = []
no_shell_ler = []
ratio_ler = []

for p in shell_p:
    shell_ler.append([pair[1] for pair in shell_p_ler_pair if pair[0] == p])

for p in no_shell_p:
    no_shell_ler.append([pair[1] for pair in no_shell_p_ler_pair if pair[0] == p])

for p in ratio_p:
    ratio_ler.append([pair[1] for pair in ratio_p_ler_pair if pair[0] == p])

# Draw boxplot
plt.boxplot(
    shell_ler,
    positions=np.arange(len(shell_p)) - 0.2,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(color='black', facecolor='blue'),
    medianprops=dict(color='black'),
    flierprops=dict(marker='.'),
)
plt.boxplot(
    no_shell_ler,
    positions=np.arange(len(no_shell_p)) + 0.2,
    widths=0.15,
    patch_artist=True,
    boxprops=dict(color='black', facecolor='red'),
    medianprops=dict(color='black'),
    flierprops=dict(marker='.'),
)

plt.boxplot(
    ratio_ler,
    positions=np.arange(len(no_shell_p)),
    widths=0.15,
    patch_artist=True,
    boxprops=dict(color='black', facecolor='orange'),
    medianprops=dict(color='black'),
    flierprops=dict(marker='.'),
)

plt.xticks(np.arange(len(shell_p)), np.around(shell_p, 4))
# Set y to log scale
plt.yscale('log')

plt.legend(
    handles=[
        Patch(facecolor='blue', label='shell'),
        Patch(facecolor='orange', label='0.36 ratio'),
        Patch(facecolor='red', label='no shell'),
    ],
    loc='upper left',
)

# Set labels and title
plt.xlabel('p for SI1000 EM')
plt.ylabel('LER')
plt.title('LER over p')


# Show the plot
plt.show()