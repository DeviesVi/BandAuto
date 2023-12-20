from defective_surface_code_adapter import Device
import os

d = 15
qdr = 0.02
cdr = 0.02

destination_dir = f'device_pool/device_d{d}_qdr{qdr}_cdr{cdr}/devices'

# Create destination directory if it does not exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)


# Parameters for generating devices
width = d
height = d
qubit_defect_rate = qdr
data_defect_rate = cdr

device_count = 100

# Generate devices
device = Device(width, height)
for i in range(device_count):
    device.add_random_defect(qubit_defect_rate, data_defect_rate)
    device.save(f'{destination_dir}/device_{device.strong_id}.pkl')