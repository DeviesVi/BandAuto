from defective_surface_code_adapter import Device
import os

ds = [15, 21, 27]
drs = [0.005, 0.01, 0.015, 0.02]

for d in ds:
    for dr in drs:
        destination_dir = f'manuscript_data/defective_devices/qubit_equal_coupler/device_d{d}_qdr{dr}_cdr{dr}/devices'

        # Create destination directory if it does not exist
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)


        # Parameters for generating devices
        width = d
        height = d
        qubit_defect_rate = dr
        coupler_defect_rate = dr

        device_count = 100

        # Generate devices
        device = Device(width, height)
        for i in range(device_count):
            device.add_random_defect(qubit_defect_rate, coupler_defect_rate)
            device.save(f'{destination_dir}/device_{device.strong_id}.pkl')