from defective_surface_code_adapter import Device, Analyzer, plot_graph
import os
from collections import defaultdict
import json

devices_paths = {
    (d, dr): f'device_pool/qubit_equal_coupler/device_d{d}_qdr{dr}_cdr{dr}/devices'
    for d in [15, 21, 27]
    for dr in [0.005, 0.01, 0.015, 0.02]
}

results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

for (d, dr), devices_path in devices_paths.items():
    for device_path in os.listdir(devices_path):
        device = Device.load(f'{devices_path}/{device_path}')
        strong_id = device.strong_id
        print(f'Analyzing device {strong_id}...')
        t_result = Analyzer.analyze_device(device, traditional_adapter=True)
        result = Analyzer.analyze_device(device)

        results[d][dr][strong_id]['Bandage'] = result.to_dict()
        results[d][dr][strong_id]['Tradition'] = t_result.to_dict()

if not os.path.exists('data/statistics'):
    os.makedirs('data/statistics')

with open('data/statistics/bandage_vs_tradition.json', 'w') as f:
    json.dump(results, f)