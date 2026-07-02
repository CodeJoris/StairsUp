"""
Module for extracting and preprocessing the OSF Stair Ambulation dataset.

This module utilizes the `gaitmap-datasets` library to load the dataset. 
It filters the internal test metadata to strictly isolate stair climbing sequences.
It extracts the pre-segmented IMU arrays and corresponding ground truth events
from the `pressure_insole_event_list_` attribute. Ground truth events (IC and TC) 
are trimmed to form complete stance phases and formatted as Nx2 arrays 
[index, foot_tag], sorted chronologically.
"""

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from gaitmap_datasets import StairAmbulationHealthy2021PerTest

def extract_ascending_stairs_data(dataset_base_path: str) -> List[Dict[str, Any]]:
    # Ensure include_pressure_data=True to fetch the ground truth
    dataset = StairAmbulationHealthy2021PerTest(
        data_folder=Path(dataset_base_path), 
        include_pressure_data=True
    )
    
    extracted_tests = []
    sensor_tags = {'right_sensor': 0, 'left_sensor': 1}
    
    print(f"\n---> [DEBUG] Loaded {len(dataset)} segmented bouts. Filtering and aligning events...")
    
    for datapoint in dataset:
        test_name = str(datapoint.group.test).lower()
        
        # Strictly filter for stair climbing (e.g., 'stair_flat_up_fast')
        if 'stair' in test_name and 'up' in test_name:
            fs = float(datapoint.sampling_rate_hz)
            
            # Use the correct attribute for discrete gait events
            events = getattr(datapoint, 'pressure_insole_event_list_', None)
            if events is None or events.empty if hasattr(events, 'empty') else not events:
                continue
            
            sensor_dict = {}
            ic_list, tc_list = [], []
            
            for sensor in ['left_sensor', 'right_sensor']:
                if sensor in datapoint.data and sensor in events:
                    sensor_df = datapoint.data[sensor]
                    
                    acc_array = sensor_df[['acc_x', 'acc_y', 'acc_z']].values
                    gyr_array = sensor_df[['gyr_x', 'gyr_y', 'gyr_z']].values
                    
                    sensor_dict[sensor] = {
                        'acc_array': acc_array,
                        'gyr_array': gyr_array
                    }
                    
                    # Extract the events for the specific foot
                    ic_raw = events[sensor]['ic'].dropna().astype(int).values
                    tc_raw = events[sensor]['tc'].dropna().astype(int).values
                    
                    # Enforce Stance Phase Boundaries (starts with IC, ends with TC)
                    if len(ic_raw) > 0 and len(tc_raw) > 0:
                        tc_raw = tc_raw[tc_raw > ic_raw[0]]
                        if len(tc_raw) > 0:
                            ic_raw = ic_raw[ic_raw < tc_raw[-1]]
                        else:
                            ic_raw = np.array([])
                    else:
                        ic_raw, tc_raw = np.array([]), np.array([])
                        
                    # Tag the arrays (0 for right, 1 for left)
                    if len(ic_raw) > 0:
                        ic_tagged = np.column_stack((ic_raw, np.full(len(ic_raw), sensor_tags[sensor])))
                        ic_list.append(ic_tagged)
                    if len(tc_raw) > 0:
                        tc_tagged = np.column_stack((tc_raw, np.full(len(tc_raw), sensor_tags[sensor])))
                        tc_list.append(tc_tagged)
                        
            # Combine chronologically
            if sensor_dict:
                ground_truth = {'ic': np.array([]), 'tc': np.array([])}
                
                if ic_list:
                    combined_ic = np.vstack(ic_list)
                    ground_truth['ic'] = combined_ic[combined_ic[:, 0].argsort()]
                if tc_list:
                    combined_tc = np.vstack(tc_list)
                    ground_truth['tc'] = combined_tc[combined_tc[:, 0].argsort()]
                    
                test_id_str = f"{datapoint.group.participant}_{datapoint.group.test}"
                
                extracted_tests.append({
                    'test_name': test_id_str,
                    'fs': fs,
                    'sensors': sensor_dict,
                    'ground_truth': ground_truth
                })
                
    print(f"---> [DEBUG] Successfully extracted {len(extracted_tests)} ascending stair sequences.")
    return extracted_tests