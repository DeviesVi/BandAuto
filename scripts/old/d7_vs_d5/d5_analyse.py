import sinter
import pickle
from typing import List, Tuple
from collections import defaultdict
from scipy import stats as st
import numpy as np
import json
import matplotlib.pyplot as plt

def calculate_ler(sample: sinter.TaskStats):
    return sample.errors/(sample.shots - sample.discards)

def LER_fit(cycle_num: np.ndarray, logi_err: np.ndarray) -> Tuple[float, float, float]:
    """Return LER, F0, Rsq"""
    lnF = [np.log(1 - 2 * e) for e in logi_err]
    k, b, r_value, _, _ = st.linregress(cycle_num, lnF)
    return (1 - np.exp(k)) / 2, np.exp(b), r_value ** 2

samples: List[sinter.TaskStats] = pickle.load(open('samples_5.pkl', 'rb'))

results = defaultdict(list)
cycles = range(2,24,2)

for sample in samples:
    results[str(sample.json_metadata['defective_qubit'])].append([sample.json_metadata['cycle'], calculate_ler(sample)])

# sort result over cycles
    
for defective_qubit, result in results.items():
    result.sort(key=lambda x: x[0])

with open('results_5.json', 'w') as f:
    json.dump(results, f)

# fit LER
LER_results = {}
for defective_qubit, result in results.items():
    cycle_num = np.array([x[0] for x in result])
    logi_err = np.array([x[1] for x in result])
    LER, F0, Rsq = LER_fit(cycle_num, logi_err)
    LER_results[defective_qubit] = {'LER': LER, 'F0': F0, 'Rsq': Rsq}

with open('LER_results_5.json', 'w') as f:
    json.dump(LER_results, f)