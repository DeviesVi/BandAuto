from defective_surface_code_adapter import Device, Adapter, Analyzer, plot_graph
import matplotlib.pyplot as plt

device1 = Device(5,5)

device1.add_ractangle_defect(5,3,1,1, clear_defect=False)
device1.add_ractangle_defect(3,5,1,1, clear_defect=False)
device1.add_ractangle_defect(5,7,1,1, clear_defect=False)
device1.add_ractangle_defect(7,5,1,1, clear_defect=False)

result1 = Analyzer.analyze_device(device1)

print(result1)

device2 = Device(5,5)
device2.add_ractangle_defect(3,3,3,3)
plot_graph(device1.graph,[])
plt.show()

result2 = Analyzer.analyze_device(device2)
print(result2)
plot_graph(device2.graph,[])
plt.show()
