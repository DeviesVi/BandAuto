from defective_surface_dode_adapter.adapter import Adapter
from defective_surface_dode_adapter.device import Device
import matplotlib.pyplot as plt
from defective_surface_dode_adapter.plot import plot_graph

qubit_defect_rate = 0.01
coupler_defect_rate = 0.01

device = Device(25, 25)
device.add_random_defect(qubit_defect_rate, coupler_defect_rate)
result = Adapter.adapt_device(device)

plt.subplot(121)
plot_graph(device.graph, [])
plt.subplot(122)
plot_graph(device.graph, result['disabled_nodes'])

plt.show()