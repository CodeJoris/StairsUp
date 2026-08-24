'''
Determine which axis is the vertical axis of the pelvis based on the orientation 
of the pelvis sensor. The vertical axis is typically aligned with the direction 
of gravity when the subject is standing upright.
'''

import os
from pathlib import Path
from typing import Optional
from gaitmap_datasets import StairAmbulationHealthy2021PerTest

import numpy as np
import pandas as pd

dataset_base_path = Path(__file__).parent.parent.parent / "data" / "osf"
COLS = ['acc_x', 'acc_y', 'acc_z']

dataset = StairAmbulationHealthy2021PerTest(
        data_folder=Path(dataset_base_path), 
        include_pressure_data=True,
        include_hip_sensor = True
    )

for datapoint in dataset:
    print(datapoint.data['hip_sensor'].keys())

    print(f'average acceleration for each axis: {np.mean(datapoint.data["hip_sensor"][COLS].values, axis=0)}')
    print('this shows that the vertical axis is the z-axis, as it has the highest average acceleration value, which corresponds to the direction of gravity when the subject is standing upright.')
    break