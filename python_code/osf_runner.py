'''
Module: osf_runner.py

This script serves as the main entry point for processing the OSF Stair 
Ambulation dataset. It extracts the target sequences, prepares the 
data, and processes it using the PyShoe toolbox. The script supports 
parallel processing to speed up the workflow.

dependencies:
- gaitmap-datasets
- pandas
- numpy
- re
- pathlib
- concurrent.futures
- tqdm
- python_code.src.preprocessing.stair_ambulation_extraction
- python_code.Toolboxes.PyShoe.pyshoe_export
'''

import pandas as pd
import numpy as np
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

from python_code.src.preprocessing.osf_surface_extraction import extract_surface_segments
from python_code.Toolboxes.PyShoe.pyshoe_export import pyshoe_process_single_file
from python_code.Toolboxes.KielMAT.kielmat_export import kielmat_process_single_file

DATA_PATH = Path(__file__).resolve().parent.parent / "data"
TARGET_MODE = "stair_up"  # CHANGE THIS: to see other surfaces 
#eg. 'stairs_down' gets you all segments with atleast 'stairs' and 'down' in the test name.

'''
surface names:
{'stair_flat_up_slow', 'staircase_up_fast', 'stair_long_flying_normal', 
'stair_long_up_fast', 'staircase_up_normal', 'stair_long_down_single_step', 
'stair_long_up_normal', 'stair_long_down_double_step', 'stair_long_down_fast', 
'stair_flat_down_fast', 'stair_long_down_slow', 'stair_long_up_double_step', 
'staircase_down_fast', 'stair_long_up_slow', 'stair_long_up_single_step', 
'staircase_up_slow', 'stair_flat_up_fast', 'stair_flat_down_slow', 
'slope_descending_normal', 'staircase_down_normal', 'slope_ascending_normal', 
'stair_flat_up_normal', 'staircase_down_slow', 'stair_long_down_normal', 
'staircase_flying_normal', 'stair_flat_down_normal'}
'''

OUTPUT_DIR = DATA_PATH / f'osf_{TARGET_MODE}'  # Change this to your desired output directory"

def process_single_test(test_data, toolbox_dir):
    """Worker function to be executed by ProcessPoolExecutor."""
    test_name = test_data['test_name']
    fs = test_data['fs']
    
    # Pre-calculate identifiers
    id_match = re.search(r'subject_(\d+)', test_name)
    subject_id = f"subj{id_match.group(1)}" if id_match else "subjXX"
    course = "OSF" 
    clip_id = re.sub(r'[^a-zA-Z0-9]', '', test_name.split('part')[-1])

    # Early Exit Check
    already_processed = True
    for detector in ['shoe', 'kielmat']:
        if not (toolbox_dir / detector / f"{subject_id}_{course}_{clip_id}.mat").exists():
            already_processed = False
            break
    if already_processed:
        return f"Skipped {test_name}"

    # Prepare Data
    sensors = test_data['sensors']
    gt = test_data['ground_truth']
    hip_acc, hip_gyr = sensors['hip_sensor']['acc_array'], sensors['hip_sensor']['gyr_array']
    acc_l, gyr_l = sensors['left_sensor']['acc_array'], sensors['left_sensor']['gyr_array']
    acc_r, gyr_r = sensors['right_sensor']['acc_array'], sensors['right_sensor']['gyr_array']
    min_len = min(len(acc_l), len(acc_r))
    
    segment_df = pd.DataFrame({
        "time": np.arange(min_len) * (1000.0 / fs),
        "acceleration_Pelvis_x": hip_acc[:min_len, 0], "acceleration_Pelvis_y": hip_acc[:min_len, 1], "acceleration_Pelvis_z": hip_acc[:min_len, 2],
        "angularVelocity_Pelvis_x": hip_gyr[:min_len, 0], "angularVelocity_Pelvis_y": hip_gyr[:min_len, 1], "angularVelocity_Pelvis_z": hip_gyr[:min_len, 2],
        "acceleration_LeftFoot_x": acc_l[:min_len, 0], "acceleration_LeftFoot_y": acc_l[:min_len, 1], "acceleration_LeftFoot_z": acc_l[:min_len, 2],
        "angularVelocity_LeftFoot_x": gyr_l[:min_len, 0], "angularVelocity_LeftFoot_y": gyr_l[:min_len, 1], "angularVelocity_LeftFoot_z": gyr_l[:min_len, 2],
        "acceleration_RightFoot_x": acc_r[:min_len, 0], "acceleration_RightFoot_y": acc_r[:min_len, 1], "acceleration_RightFoot_z": acc_r[:min_len, 2],
        "angularVelocity_RightFoot_x": gyr_r[:min_len, 0], "angularVelocity_RightFoot_y": gyr_r[:min_len, 1], "angularVelocity_RightFoot_z": gyr_r[:min_len, 2],
    })
    
    ic, tc = gt['ic'], gt['tc']
    true_hs_r = (ic[ic[:, 1] == 0][:, 0] * (1000.0 / fs)) if len(ic) > 0 else np.array([])
    true_hs_l = (ic[ic[:, 1] == 1][:, 0] * (1000.0 / fs)) if len(ic) > 0 else np.array([])
    true_fo_r = (tc[tc[:, 1] == 0][:, 0] * (1000.0 / fs)) if len(tc) > 0 else np.array([])
    true_fo_l = (tc[tc[:, 1] == 1][:, 0] * (1000.0 / fs)) if len(tc) > 0 else np.array([])

    pyshoe_result = pyshoe_process_single_file(
        segment_df, True, course, subject_id, clip_id, 
        true_hs_r, true_fo_r, 
        true_hs_l, true_fo_l, 
        toolbox_dir, fs
    )
    kielmat_result = kielmat_process_single_file(
        segment_df, course, subject_id, clip_id, 
        true_hs_r, true_fo_r, 
        true_hs_l, true_fo_l, 
        toolbox_dir, fs
    )

    return f"{test_name} => PyShoe [{pyshoe_result}] | KielMAT [{kielmat_result}]"

if __name__ == "__main__":
    NILSPOD_DATA_PATH = DATA_PATH / "osf"
    
    if NILSPOD_DATA_PATH.exists():
        ascending_tests = extract_surface_segments(str(NILSPOD_DATA_PATH), TARGET_MODE)
        
        # Use partial to pass the toolbox_dir argument
        worker_func = partial(process_single_test, toolbox_dir=OUTPUT_DIR)
        
        # Execute in parallel
        print("\nStarting Parallel Processing...")
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(worker_func, ascending_tests), total=len(ascending_tests)))