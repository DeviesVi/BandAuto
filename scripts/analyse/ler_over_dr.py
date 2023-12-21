device_paths = [
    'device_pool/device_d21_qdr0.02_cdr0.02/devices',
    'device_pool/device_d21_qdr0.015_cdr0.015/devices',
    'device_pool/device_d21_qdr0.01_cdr0.01/devices',
    'device_pool/device_d21_qdr0.005_cdr0.005/devices',
]

sample_paths = [
    'device_pool/device_d21_qdr0.02_cdr0.02/samples_over_holding_options_p0.002',
    'device_pool/device_d21_qdr0.015_cdr0.015/samples_over_specified_cycle',
    'device_pool/device_d21_qdr0.01_cdr0.01/samples_over_specified_cycle',
    'device_pool/device_d21_qdr0.005_cdr0.005/samples_over_specified_cycle',
]

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import os
import sinter
import numpy as np
import pickle
from typing import List, Tuple
from defective_surface_code_adapter import Device, Analyzer

from utils import calculate_ler

results = {
    0.02: [],
    0.015: [],
    0.01: [],
    0.005: [],
}

for i, device_path in enumerate(device_paths):
    for device_file in os.listdir(device_path):
        device = Device.load(f'{device_path}/{device_file}')
        samples: List[sinter.TaskStats] = pickle.load(open(f'{sample_paths[i]}/samples_{device.strong_id}.pkl', 'rb'))
        # Calculate min LER for each device
        min_ler = 1
        for sample in samples:
            if sample.json_metadata['holding_cycle_option'] == 'SPECIFIED':
                min_ler = min(min_ler, calculate_ler(sample))
        results[device.qubit_defect_rate].append(min_ler)


# Draw boxplot
plt.boxplot(
    [results[0.02], results[0.015], results[0.01], results[0.005]],
    labels=[0.02, 0.015, 0.01, 0.005],
    patch_artist=True,
    boxprops=dict(facecolor='lightblue'),
    medianprops=dict(color='black'),
    flierprops=dict(marker='.'),
)

# set y to log scale
plt.yscale('log')

plt.title('Minimum LER vs Defect Rate')
plt.xlabel('Defect Rate')
plt.ylabel('Minimum LER')
plt.show()


        
