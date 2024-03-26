from defective_surface_code_adapter import Device
import os

ds = [21]
drs = [0.02, 0.04, 0.06, 0.08]

for d in ds:
    for dr in drs:
        destination_dir = f'device_pool/dirty_area/device_d{d}_qdr{dr}_cdr{dr}/devices'

        # Create destination directory if it does not exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)


        # Parameters for generating devices
        width = d
        height = d
        qubit_defect_rate = dr
        data_defect_rate = dr

        device_count = 100

        # Generate devices
        device = Device(width, height)
        for i in range(device_count):
            device.add_random_defect_in_area(11, 11, 31, 31, qubit_defect_rate, data_defect_rate)
            device.save(f'{destination_dir}/device_{device.strong_id}.pkl')