import sys
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.io import savemat

PYSHOE_DIR = Path(__file__).resolve().parent
if str(PYSHOE_DIR) not in sys.path:
    sys.path.insert(0, str(PYSHOE_DIR))

from ins_tools.INS import INS

# Necessary data for pyshoe algorithm (order matters)
RIGHT_FOOT_COLS = ['acceleration_RightFoot_x', 'acceleration_RightFoot_y', 'acceleration_RightFoot_z', 'angularVelocity_RightFoot_x', 'angularVelocity_RightFoot_y', 'angularVelocity_RightFoot_z']
LEFT_FOOT_COLS = ['acceleration_LeftFoot_x', 'acceleration_LeftFoot_y', 'acceleration_LeftFoot_z', 'angularVelocity_LeftFoot_x', 'angularVelocity_LeftFoot_y', 'angularVelocity_LeftFoot_z']
TARGET_COLS = ["time"] + RIGHT_FOOT_COLS + LEFT_FOOT_COLS

SAMPLING_FREQUENCY = 60
DETECTORS = ['shoe', 'ared', 'amvd', 'mbgtd']
SPECS = {  # G values are sensor/walking surface dependent more at https://github.com/utiasSTARS/pyshoe/tree/master
    'shoe': {"G":2e8},
    'ared': {"G":2.0},
    'amvd': {"G":7},
    'mbgtd': {"G":10},
}


def pyshoe_process_single_file(
    segment: pd.DataFrame, 
    course: str, 
    id: str, 
    clip_id: int,  # Added to separate staircase events
    y_HS_true: pd.DataFrame | np.ndarray, 
    y_FO_true: pd.DataFrame | np.ndarray, 
    data_path: Path
) -> str:
    """
    Processes a single trial dataset across all four zero-velocity update (ZUPT) 
    detectors implemented in PyShoe.

    This function extracts IMU data for both left and right feet (3-axis accelerometer 
    and 3-axis gyroscope data), applies the 'shoe', 'ared', 'amvd', and 'mbgtd' state 
    detectors, computes bilateral heel strike and foot off events, and saves separate 
    MATLAB .mat summaries for each unique detector.

    Parameters:
    -----------
    segment : pd.DataFrame
        The data source for the trial. A pre-sliced Pandas DataFrame containing the target 
        time frame. Must contain a 'time' index alongside the bilateral 
        acceleration and angular velocity columns.
        
    course : str
        The name of the course or environment directory (e.g., 'course_1').
        
    id : str
        The subject identifier string matching the directory layout (e.g., 'subj_01').
        
    y_HS_true : pd.DataFrame or pd.Series or np.ndarray
        The ground truth heel strike annotations for this specific segment window.
        
    y_FO_true : pd.DataFrame or pd.Series or np.ndarray
        The ground truth foot off annotations for this specific segment window.
        
    data_path : pathlib.Path
        The base Path object pointing to the global data directory where outputs 
        will be structured across subfolders named after each active detector.

    Returns:
    --------
    str
        A status summary string confirming completion of missing detectors, a skip message 
        if all four detectors are already found cached on disk, or an error description.
    """
    # Look at the disk first to see which detectors actually need to be run
    detectors_to_run = []
    for detector_name in DETECTORS:
        out_file = data_path / "toolbox1" / detector_name / f'{id}_{course}_{clip_id}.mat'
        if not out_file.exists():
            detectors_to_run.append((detector_name, out_file))
            
    # If the list is empty, every single detector has already been processed. 
    if not detectors_to_run:
        return f"Skipped {id}_{course} (Already completely processed)"

    try:
        imu_left = segment[LEFT_FOOT_COLS].to_numpy() 
        imu_right = segment[RIGHT_FOOT_COLS].to_numpy() # PyShoe takes in an N x 6 numpy array
        time = segment["time"].to_numpy().squeeze()

        # loop detectors
        for detector_name, out_file in detectors_to_run:
            G_val = SPECS[detector_name]['G']

            # Left Foot
            ins_left = INS(imu_left, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/SAMPLING_FREQUENCY) 
            ins_left.baseline(W=5, G=G_val, detector=detector_name)
            steps_left = ins_left.zv
            
            padded_left = np.insert(steps_left, 0, False).astype(int)
            diff_left = np.diff(padded_left)
            HS_indices_left = np.where(diff_left == 1)[0]
            FO_indices_left = np.where(diff_left == -1)[0]
            # Case where foot started stationary
            if steps_left[0]:
                HS_indices_left = HS_indices_left[1:]
                FO_indices_left = FO_indices_left[1:]
            HS_times_left = time[HS_indices_left]
            FO_times_left = time[FO_indices_left]

            # Right Foot
            ins_right = INS(imu_right, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/60) 
            ins_right.baseline(W=5, G=G_val, detector=detector_name)
            steps_right = ins_right.zv 
            
            padded_right = np.insert(steps_right, 0, False).astype(int)
            diff_right = np.diff(padded_right)
            HS_indices_right = np.where(diff_right == 1)[0]
            FO_indices_right = np.where(diff_right == -1)[0]
            if steps_right[0]:
                HS_indices_right = HS_indices_right[1:]
                FO_indices_right = FO_indices_right[1:]
            HS_times_right = time[HS_indices_right]
            FO_times_right = time[FO_indices_right]

            # Combine
            all_HS_times = np.concatenate((HS_times_left, HS_times_right))
            all_FO_times = np.concatenate((FO_times_left, FO_times_right))
            
            y_HS_pred = np.zeros((len(all_HS_times), 2))
            y_FO_pred = np.zeros((len(all_FO_times), 2))
            y_HS_pred[:,0] = all_HS_times
            y_FO_pred[:,0] = all_FO_times

            results = {
                'y_HS': y_HS_true,         # true
                'y_FO': y_FO_true,
                'y_hat_HS': y_HS_pred,     # pred
                'y_hat_FO': y_FO_pred
            }

            # Save specific detector output
            out_file.parent.mkdir(parents=True, exist_ok=True)
            savemat(out_file, {'results': results})

        return f"{id}_{course}_{clip_id}.mat Success"
    
    except Exception as e:
        return f"Error: {str(e)}"

