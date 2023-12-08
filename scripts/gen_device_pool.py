from defective_surface_code_adapter import Device

destination_dir = 'device_pool/devices/'


# Parameters for generating devices
width = 21
height = 21
qubit_defect_rate = 0.02
data_defect_rate = 0.02

device_count = 100

# Generate devices
device = Device(width, height)
for i in range(device_count):
    device.add_random_defect(qubit_defect_rate, data_defect_rate)
    device.save(f'{destination_dir}device_{device.strong_id}.pkl')