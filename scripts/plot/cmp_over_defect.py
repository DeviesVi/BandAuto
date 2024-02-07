import matplotlib.pyplot as plt
import pickle
import sinter
from typing import List
from utils import calculate_ler
import matplotlib.pyplot as plt


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



plt.figure(figsize=(8, 6), dpi=100)
for s in ['0', '+']:
    if s == '0':
        plt.plot(range(1, 4), results[s][0], label=f'Bandage |{s}>', marker='o', color='C0')
        plt.plot(range(1, 4), results[s][1], label=f'Tradition |{s}>', marker='o', linestyle='--', color='C0')
    else:
        plt.plot(range(1, 4), results[s][0], label=f'Bandage |{s}>', marker='^', color='C2')
        plt.plot(range(1, 4), results[s][1], label=f'Tradition |{s}>', marker='^', linestyle='--', color='C2')
    
    # Set x-ticks to integers
plt.xticks(range(1, 4))
plt.xlabel('Defective Data Qubits')
plt.ylabel('LER')
plt.legend()
plt.grid()

plt.savefig('cmp_defect.pdf', format='pdf')
plt.show()