from __future__ import annotations

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
    'shoe': {"G":2.45e8},
    'ared': {"G":2.0},
    'amvd': {"G":7},
    'mbgtd': {"G":43},
}

def clean_gait_events(hs_times: np.ndarray, fo_times: np.ndarray, sampling_freq: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cleans raw PyShoe events by dropping incomplete boundary steps and 
    enforcing a minimum stance time to void false-positive flickers.
    """
    # 1. Drop lone initial Foot Off (mid-swing start)
    if len(fo_times) > 0 and len(hs_times) > 0:
        if fo_times[0] < hs_times[0]:
            fo_times = fo_times[1:]
            
    # 2. Drop lone trailing Heel Strike (stance-phase end)
    if len(hs_times) > 0 and len(fo_times) > 0:
        if hs_times[-1] > fo_times[-1]:
            hs_times = hs_times[:-1]

    # 3. Enforce Minimum Stance Time
    MIN_STANCE_FRAMES = 15  # Adjust if needed
    MIN_STANCE_MS = (MIN_STANCE_FRAMES / sampling_freq) * 1000.0
    
    if len(hs_times) == len(fo_times) and len(hs_times) > 0:
        stance_durations = fo_times - hs_times
        valid_mask = stance_durations >= MIN_STANCE_MS
        hs_times = hs_times[valid_mask]
        fo_times = fo_times[valid_mask]
        
    return hs_times, fo_times


def pyshoe_process_single_file(
    segment: pd.DataFrame, 
    course: str, 
    id: str, 
    clip_id: int,
    true_hs_r: pd.DataFrame | np.ndarray, 
    true_fo_r: pd.DataFrame | np.ndarray, 
    true_hs_l: pd.DataFrame | np.ndarray, 
    true_fo_l: pd.DataFrame | np.ndarray, 
    data_path: Path,
    fs: float
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
        
    true_hs_r : pd.DataFrame or pd.Series or np.ndarray
        The ground truth heel strike annotations for this specific segment window.
        
    true_fo_r : pd.DataFrame or pd.Series or np.ndarray
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
    detectors_to_run = []
    for detector_name in DETECTORS:
        out_file = data_path / detector_name / f'{id}_{course}_{clip_id}.mat'
        if not out_file.exists():
            detectors_to_run.append((detector_name, out_file))
            
    if not detectors_to_run:
        return f"Skipped {id}_{course} (Already completely processed)"

    try:
        imu_left = segment[LEFT_FOOT_COLS].to_numpy() 
        imu_right = segment[RIGHT_FOOT_COLS].to_numpy() 
        time = segment["time"].to_numpy().squeeze()

        for detector_name, out_file in detectors_to_run:
            G_val = SPECS[detector_name]['G']

            # Left Foot (Using dynamic fs)
            ins_left = INS(imu_left, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/fs) 
            ins_left.baseline(W=5, G=G_val, detector=detector_name)
            steps_left = ins_left.zv
            
            padded_left = np.insert(steps_left, 0, False).astype(int)
            diff_left = np.diff(padded_left)
            HS_indices_left = np.where(diff_left == 1)[0]
            FO_indices_left = np.where(diff_left == -1)[0]
            if steps_left[0] and len(HS_indices_left) > 0:
                HS_indices_left = HS_indices_left[1:]

            HS_times_left = time[HS_indices_left]
            FO_times_left = time[FO_indices_left]

            # Right Foot (Using dynamic fs)
            ins_right = INS(imu_right, sigma_a=0.00098, sigma_w=8.7266463e-5, T=1.0/fs) 
            ins_right.baseline(W=5, G=G_val, detector=detector_name)
            steps_right = ins_right.zv 
            
            padded_right = np.insert(steps_right, 0, False).astype(int)
            diff_right = np.diff(padded_right)
            HS_indices_right = np.where(diff_right == 1)[0]
            FO_indices_right = np.where(diff_right == -1)[0]
            if steps_right[0] and len(HS_indices_right) > 0:
                HS_indices_right = HS_indices_right[1:]

            HS_times_right = time[HS_indices_right]
            FO_times_right = time[FO_indices_right]

            # Clean the raw events for both feet using dynamic fs
            HS_times_left, FO_times_left = clean_gait_events(HS_times_left, FO_times_left, fs)
            HS_times_right, FO_times_right = clean_gait_events(HS_times_right, FO_times_right, fs)

            def tag_foot(times, label):
                if len(times) == 0: return np.empty((0, 2))
                return np.column_stack((times, np.full(len(times), label)))

            pred_hs_r = tag_foot(HS_times_right, 0)
            pred_hs_l = tag_foot(HS_times_left, 1)
            pred_fo_r = tag_foot(FO_times_right, 0)
            pred_fo_l = tag_foot(FO_times_left, 1)

            y_HS_pred = np.vstack((pred_hs_r, pred_hs_l)) if len(pred_hs_r) or len(pred_hs_l) else np.empty((0, 2))
            y_FO_pred = np.vstack((pred_fo_r, pred_fo_l)) if len(pred_fo_r) or len(pred_fo_l) else np.empty((0, 2))
            
            if len(y_HS_pred): y_HS_pred = y_HS_pred[y_HS_pred[:, 0].argsort()]
            if len(y_FO_pred): y_FO_pred = y_FO_pred[y_FO_pred[:, 0].argsort()]

            true_hs_r_tagged = tag_foot(true_hs_r, 0)
            true_hs_l_tagged = tag_foot(true_hs_l, 1)
            true_fo_r_tagged = tag_foot(true_fo_r, 0)
            true_fo_l_tagged = tag_foot(true_fo_l, 1)

            y_HS_true = np.vstack((true_hs_r_tagged, true_hs_l_tagged)) if len(true_hs_r_tagged) or len(true_hs_l_tagged) else np.empty((0, 2))
            y_FO_true = np.vstack((true_fo_r_tagged, true_fo_l_tagged)) if len(true_fo_r_tagged) or len(true_fo_l_tagged) else np.empty((0, 2))

            if len(y_HS_true): y_HS_true = y_HS_true[y_HS_true[:, 0].argsort()]
            if len(y_FO_true): y_FO_true = y_FO_true[y_FO_true[:, 0].argsort()]

            results = {
                'y_HS': y_HS_true,         
                'y_FO': y_FO_true,
                'y_hat_HS': y_HS_pred,     
                'y_hat_FO': y_FO_pred
            }

            out_file.parent.mkdir(parents=True, exist_ok=True)
            savemat(out_file, {'results': results})

        return f"{id}_{course}_{clip_id}.mat Success"
    
    except Exception as e:
        return f"Error on {id}_{course}_{clip_id}: {str(e)}"