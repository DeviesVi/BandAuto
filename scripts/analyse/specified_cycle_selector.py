import matplotlib.pyplot as plt
import json
import os
from defective_surface_code_adapter import Device, Analyzer
import pickle
import sinter
from typing import List, Callable, Any, Dict, Tuple
import numpy as np
from data_dir import device_dir, sample_dir

def calculate_ler(sample: sinter.TaskStats):
    return sample.errors/(sample.shots - sample.discards)

result_dir = 'device_pool\device_d21_qdr0.02_cdr0.02\cycle_selected_by_p0.002.json'

min_ler_pos: Dict[str, int] = {}

filter_funcs: Dict[Tuple[str, str], Callable[['sinter.TaskStats'], Any]] = {
    ("SPECIFIED", "0"): lambda stat: stat.json_metadata["initial_state"] == "0" and stat.json_metadata["holding_cycle_option"] == "SPECIFIED",
    ("SPECIFIED", "+"): lambda stat: stat.json_metadata["initial_state"] == "+" and stat.json_metadata["holding_cycle_option"] == "SPECIFIED",
}

# Average over zero and plus initial states
for file in os.listdir(device_dir):
    device = Device.load(f"{device_dir}{file}")
    samples_path = f"{sample_dir}/samples_{device.strong_id}.pkl"
    samples: List[sinter.TaskStats] = pickle.load(open(samples_path, "rb"))

    # Record ler and specified_holding_cycle for each sample
    ler_by_cycle: Dict[int, List[float]] = {}
    for sample in samples:
        if filter_funcs[("SPECIFIED", "0")](sample) or filter_funcs[("SPECIFIED", "+")](sample):
            cycle = sample.json_metadata["specified_holding_cycle"]
            if cycle not in ler_by_cycle:
                ler_by_cycle[cycle] = []
            ler_by_cycle[cycle].append(calculate_ler(sample))

    # Calculate average ler for each cycle
    avg_ler_by_cycle = {}
    for cycle, lers in ler_by_cycle.items():
        avg_ler_by_cycle[cycle] = sum(lers)/len(lers)
    
    # Find minimum avg ler cycle
    min_ler_cycle = None
    for cycle, avg_ler in avg_ler_by_cycle.items():
        if min_ler_cycle is None:
            min_ler_cycle = cycle
        elif avg_ler < avg_ler_by_cycle[min_ler_cycle]:
            min_ler_cycle = cycle

    # Record minimum avg ler cycle
    min_ler_pos[device.strong_id] = min_ler_cycle

# Dump result to file
with open(result_dir, 'w') as f:
    json.dump(min_ler_pos, f)