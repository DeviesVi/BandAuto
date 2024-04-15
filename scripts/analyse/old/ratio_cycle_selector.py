from math import floor
import matplotlib.pyplot as plt
import json
import os
from defective_surface_code_adapter import Device, Analyzer
import pickle
import sinter
from typing import List, Callable, Any, Dict, Tuple
import numpy as np
from scripts.analyse.old.data_dir import device_dir, sample_dir

def calculate_ler(sample: sinter.TaskStats):
    return sample.errors/(sample.shots - sample.discards)

ratio = 0.36

result_dir = f'device_pool/device_d21_qdr0.02_cdr0.02/cycle_selected_by_ratio{ratio}_floor.json'

ratio_shell_dict: Dict[str, int] = {}

for file in os.listdir(device_dir):
    device = Device.load(f"{device_dir}/{file}")
    ana_result = Analyzer.analyze_device(device)
    avg_weight = ana_result.stabilizer_statistics['avg_stabilizer_weight']
    ratio_shell = floor(avg_weight * ratio)
    ratio_shell_dict[device.strong_id] = ratio_shell

# Dump result to file
with open(result_dir, 'w') as f:
    json.dump(ratio_shell_dict, f)