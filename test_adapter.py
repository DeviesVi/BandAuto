from defective_surface_code_adapter.adapter import Adapter
from defective_surface_code_adapter.device import Device
import matplotlib.pyplot as plt
from defective_surface_code_adapter.plot import plot_graph

qubit_defect_rate = 0.005
coupler_defect_rate = 0.005

device = Device(25, 25)
device.add_random_defect(qubit_defect_rate, coupler_defect_rate)
device.save('device.pkl')
# device = Device.load('device.pkl')
result = Adapter.adapt_device(device)

plt.subplot(121)
plot_graph(device.graph, [])
plt.subplot(122)

super_stabilizers = [stabilizer for stabilizer in result.stabilizers if len(stabilizer) > 1]
print(super_stabilizers)

super_stabilizer_nodes = [node for stabilizer in super_stabilizers for node in stabilizer]

logical_operator_nodes = result.logical_x_data_qubits + result.logical_z_data_qubits

plot_graph(device.graph, result.disabled_nodes, logical_operator_nodes = logical_operator_nodes)


plt.show()