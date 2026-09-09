"""
Module: src.preprocessing.osf_surface_extraction
Description: 
    Extracts and preprocesses the OSF Stair Ambulation dataset. 
    It filters the internal test metadata to strictly isolate specific 
    stair climbing sequences. Ground truth events (IC and TC) are 
    extracted from the pressure insoles, trimmed to form complete 
    stance phases, and formatted as chronologically sorted arrays.

Dependencies:
    - numpy
    - pandas
    - pathlib
    - gaitmap-datasets
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path
from gaitmap_datasets import StairAmbulationHealthy2021PerTest


def is_target_surface(test_name: str, surface: str) -> bool:
    """
    Evaluates whether a test name matches the target surface criteria.

    Parameters
    ----------
    test_name : str
        The full test name from the dataset group.
    surface : str
        The target surface string containing keywords separated by underscores.

    Returns
    -------
    bool
        True if all target keywords are present in the test name, False otherwise.
    """
    parts = surface.split('_')
    return all(part in test_name for part in parts)


def enforce_stance_boundaries(
    ic_raw: np.ndarray, 
    tc_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trims initial and terminal contact events to ensure strict stance phase sequences.
    
    A valid stance phase must begin with an Initial Contact (IC) and end with 
    a Terminal Contact (TC).

    Parameters
    ----------
    ic_raw : np.ndarray
        Raw array of Initial Contact indices.
    tc_raw : np.ndarray
        Raw array of Terminal Contact indices.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the trimmed (ic, tc) arrays.
    """
    if len(ic_raw) > 0 and len(tc_raw) > 0:
        # Ensure the first TC happens after the first IC
        tc_trimmed = tc_raw[tc_raw > ic_raw[0]]
        
        if len(tc_trimmed) > 0:
            # Ensure the last IC happens before the last TC
            ic_trimmed = ic_raw[ic_raw < tc_trimmed[-1]]
            return ic_trimmed, tc_trimmed
            
    return np.array([]), np.array([])


def extract_sensor_arrays(sensor_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Extracts accelerometer and gyroscope arrays from a sensor dataframe and applies unit conversions.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Dataframe containing raw 6-axis IMU data.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing 'acc_array' (m/s^2) and 'gyr_array' (rad/s).
    """
    acc_array = sensor_df[['acc_x', 'acc_y', 'acc_z']].values
    # Convert gyroscope data from degrees/sec to radians/sec
    gyr_array = np.deg2rad(sensor_df[['gyr_x', 'gyr_y', 'gyr_z']].values)
    
    return {
        'acc_array': acc_array,
        'gyr_array': gyr_array
    }


def aggregate_ground_truth(
    ic_list: List[np.ndarray], 
    tc_list: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Combines tagged event lists from multiple sensors and sorts them chronologically.

    Parameters
    ----------
    ic_list : List[np.ndarray]
        List of Initial Contact arrays (N x 2: [index, tag]).
    tc_list : List[np.ndarray]
        List of Terminal Contact arrays (N x 2: [index, tag]).

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary mapping 'ic' and 'tc' to chronologically sorted Nx2 arrays.
    """
    ground_truth = {'ic': np.array([]), 'tc': np.array([])}
    
    if ic_list:
        combined_ic = np.vstack(ic_list)
        ground_truth['ic'] = combined_ic[combined_ic[:, 0].argsort()]
        
    if tc_list:
        combined_tc = np.vstack(tc_list)
        ground_truth['tc'] = combined_tc[combined_tc[:, 0].argsort()]
        
    return ground_truth


def extract_surface_segments(
    dataset_base_path: str, 
    surface: str
) -> List[Dict[str, Any]]:
    """
    Iterates through the dataset to locate, extract, and align IMU segments 
    and ground truth events for a specific walking surface.

    Parameters
    ----------
    dataset_base_path : str
        The local path to the root folder of the gait dataset.
    surface : str
        The surface condition to filter by (e.g., 'stair_flat_up_fast').

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing formatted test data, including 
        sampling frequency, nested sensor arrays, and ground truth event arrays.
    """
    dataset = StairAmbulationHealthy2021PerTest(
        data_folder=Path(dataset_base_path), 
        include_pressure_data=True,
        include_hip_sensor=True
    )

    extracted_tests = []
    sensor_tags = {'right_sensor': 0, 'left_sensor': 1}
    
    print(f"\n---> [DEBUG] Loaded {len(dataset)} segmented bouts. Filtering and aligning events...")
    
    for datapoint in dataset:
        test_name = str(datapoint.group.test).lower()
        
        if not is_target_surface(test_name, surface):
            continue
            
        fs = float(datapoint.sampling_rate_hz)
        events = getattr(datapoint, 'pressure_insole_event_list_', None)
        
        if events is None or (hasattr(events, 'empty') and events.empty) or not events:
            continue
        
        sensor_dict = {}
        ic_list, tc_list = [], []
        
        for sensor in ['left_sensor', 'right_sensor', 'hip_sensor']:
            if sensor not in datapoint.data:
                continue
                
            sensor_dict[sensor] = extract_sensor_arrays(datapoint.data[sensor])
            
            # Extract events for the specific foot
            if sensor in events:
                ic_raw = events[sensor]['ic'].dropna().astype(int).values
                tc_raw = events[sensor]['tc'].dropna().astype(int).values
            else:
                ic_raw, tc_raw = np.array([]), np.array([])
            
            ic_trimmed, tc_trimmed = enforce_stance_boundaries(ic_raw, tc_raw)
                
            # Tag the arrays (0 for right, 1 for left)
            if len(ic_trimmed) > 0:
                ic_tagged = np.column_stack((ic_trimmed, np.full(len(ic_trimmed), sensor_tags[sensor])))
                ic_list.append(ic_tagged)
            if len(tc_trimmed) > 0:
                tc_tagged = np.column_stack((tc_trimmed, np.full(len(tc_trimmed), sensor_tags[sensor])))
                tc_list.append(tc_tagged)
                    
        if sensor_dict:
            ground_truth = aggregate_ground_truth(ic_list, tc_list)
            test_id_str = f"{datapoint.group.participant}_{datapoint.group.test}"
            
            extracted_tests.append({
                'test_name': test_id_str,
                'fs': fs,
                'sensors': sensor_dict,
                'ground_truth': ground_truth
            })
                
    print(f"---> [DEBUG] Successfully extracted {len(extracted_tests)} ascending stair sequences.")
    return extracted_tests