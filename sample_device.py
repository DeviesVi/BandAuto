from defective_surface_code_adapter import (
    Device,
    Adapter,
    Analyzer,
    plot_graph,
)

import matplotlib.pyplot as plt
from math import log

device = Device(11, 11)

defective_nodes = [
    (9, 1),
    (17, 1),
    (12, 2),
    (5, 7),
    (20, 10),
    (9, 11),
    (7, 13),
    (16, 14),
    (9, 15),
    (1, 21),
    (17, 21),
]

defective_edges = [
    ((12, 4), (13, 5)),
    ((16, 8), (17, 9)),
    ((1, 11), (0, 12)),
    ((16, 14), (15, 15)),
]

for node in defective_nodes:
    device.graph.nodes[node]["defective"] = True

for edge in defective_edges:
    device.graph.edges[edge]["defective"] = True

result = Adapter.adapt_device(device)

plt.subplot(121)
plot_graph(device.graph, [])
plt.subplot(122)

super_stabilizers = [stabilizer for stabilizer in result.stabilizers if len(stabilizer) > 1]
print(super_stabilizers)

super_stabilizer_nodes = [node for stabilizer in super_stabilizers for node in stabilizer]

logical_operator_nodes = result.logical_x_data_qubits + result.logical_z_data_qubits

ana_result = Analyzer.analyze_device(device)

print(ana_result)
print(log(ana_result.x_shortest_path_count))
print(log(ana_result.z_shortest_path_count))

plot_graph(device.graph, result.disabled_nodes, logical_operator_nodes=result.logical_x_data_qubits)

plt.show()