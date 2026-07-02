#%% Imports
import pandas as pd
import numpy as np
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

from python_code.src.preprocessing.stair_ambulation_extraction import extract_ascending_stairs_data
from python_code.Toolboxes.PyShoe.pyshoe_export import pyshoe_process_single_file

DATA_PATH = Path(__file__).resolve().parent.parent / "data"
TOOLBOX_DIR = DATA_PATH / "ambulation_out"

def process_single_test(test_data, toolbox_dir):
    """Worker function to be executed by ProcessPoolExecutor."""
    test_name = test_data['test_name']
    fs = test_data['fs']
    
    # Pre-calculate identifiers
    id_match = re.search(r'subject_(\d+)', test_name)
    subject_id = f"subj{id_match.group(1)}" if id_match else "subjXX"
    course = "OSF" 
    clip_id = "clip_" + re.sub(r'[^a-zA-Z0-9]', '', test_name.split('part')[-1])

    # Early Exit Check
    already_processed = True
    for detector in ['shoe', 'ared', 'amvd', 'mbgtd']:
        if not (toolbox_dir / detector / f"{subject_id}_{course}_{clip_id}.mat").exists():
            already_processed = False
            break
    if already_processed:
        return f"Skipped {test_name}"

    # Prepare Data
    sensors = test_data['sensors']
    gt = test_data['ground_truth']
    acc_l, gyr_l = sensors['left_sensor']['acc_array'], sensors['left_sensor']['gyr_array']
    acc_r, gyr_r = sensors['right_sensor']['acc_array'], sensors['right_sensor']['gyr_array']
    min_len = min(len(acc_l), len(acc_r))
    
    segment_df = pd.DataFrame({
        "time": np.arange(min_len) * (1000.0 / fs),
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

    return pyshoe_process_single_file(
        segment_df, course, subject_id, clip_id, true_hs_r, true_fo_r, true_hs_l, true_fo_l, toolbox_dir, fs
    )

if __name__ == "__main__":
    NILSPOD_DATA_PATH = DATA_PATH / "stair_ambulation_dataset"
    
    if NILSPOD_DATA_PATH.exists():
        ascending_tests = extract_ascending_stairs_data(str(NILSPOD_DATA_PATH))
        
        # Use partial to pass the toolbox_dir argument
        worker_func = partial(process_single_test, toolbox_dir=TOOLBOX_DIR)
        
        # Execute in parallel
        print("\nStarting Parallel Processing...")
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(worker_func, ascending_tests), total=len(ascending_tests)))