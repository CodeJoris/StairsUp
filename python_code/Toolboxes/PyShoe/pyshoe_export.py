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

DETECTORS = ['shoe', 'ared', 'amvd', 'mbgtd']
SPECS = {  # G values are sensor/walking surface dependent more at https://github.com/utiasSTARS/pyshoe/tree/master
    'shoe': {"G":15000},
    'ared': {"G":2.0},
    'amvd': {"G":7},
    'mbgtd': {"G":43},
}

def clean_gait_events(hs_times: np.ndarray, fo_times: np.ndarray, sampling_freq: float) -> tuple[np.ndarray, np.ndarray]:
    """Processes a single gait segment to extract step timings using PyShoe.

        Executes the zero-velocity detection pipeline, calculates signal noise floor
        dynamically, detects gait events, and exports the results to a .mat file 
        compatible with the GaitAnalysisPipeline.

        Args:
            segment (pd.DataFrame): DataFrame containing columns for both feet.
            course (str): Identifier for the experimental course.
            id (str): Subject identifier.
            clip_id (int): Segment clip index.
            true_hs_r/l (np.ndarray): Ground truth HS timestamps for right/left foot.
            true_fo_r/l (np.ndarray): Ground truth FO timestamps for right/left foot.
            data_path (Path): Directory path to save the resulting .mat file.
            fs (float): Sampling frequency of the segment.

        Returns:
            str: A status message indicating success or failure of the processing.
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
    MIN_STANCE_MS = 300
    
    if len(hs_times) == len(fo_times) and len(hs_times) > 0:
        stance_durations = fo_times - hs_times
        valid_mask = stance_durations >= MIN_STANCE_MS
        hs_times = hs_times[valid_mask]
        fo_times = fo_times[valid_mask]
        
    return hs_times, fo_times

def estimate_noise_from_midstance(imu_data: np.ndarray, fs: float, window_duration_sec: float = 0.1) -> tuple[float, float]:
    """
    Estimates the baseline sensor noise (sigma_a, sigma_w) by isolating the 
    quietest midstance (minimum variance window) in the provided IMU data.
    
    Parameters
    -----------
    imu_data : np.ndarray
        N x 6 array containing [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    fs : float
        Sampling frequency of the data in Hz.
    window_duration_sec : float
        Duration of the sliding window in seconds (0.1s is typical for midstance).
        
    Returns
    --------
    tuple : [float, float]
        Estimated accelerometer (sigma_a) and gyroscope (sigma_w) noise standard deviations.
    """
    # Calculate window size in samples (minimum of 5 to ensure valid stats)
    window_size = max(5, int(fs * window_duration_sec))
    
    # Split acceleration and gyroscope data
    acc = imu_data[:, 0:3]
    gyr = imu_data[:, 3:6]
    
    # Calculate the magnitude of the acceleration vector
    acc_mag = np.linalg.norm(acc, axis=1)
    
    # Calculate rolling variance of acceleration magnitude to find the quietest period
    acc_mag_series = pd.Series(acc_mag)
    rolling_var = acc_mag_series.rolling(window=window_size).var()
    
    # Find the end index of the quietest window
    quietest_end_idx = rolling_var.idxmin()
    
    # Fallback if array is too short and returns NaN
    if pd.isna(quietest_end_idx):
        sigma_a = np.mean(np.std(acc, axis=0))
        sigma_w = np.mean(np.std(gyr, axis=0))
    else:
        quietest_start_idx = int(quietest_end_idx) - window_size + 1
        
        # Extract the static window for both sensors
        static_acc = acc[quietest_start_idx : int(quietest_end_idx) + 1]
        static_gyr = gyr[quietest_start_idx : int(quietest_end_idx) + 1]
        
        # Calculate mean of standard deviations across the 3 axes
        sigma_a = np.mean(np.std(static_acc, axis=0))
        sigma_w = np.mean(np.std(static_gyr, axis=0))
        
    # Enforce a minimum noise floor to prevent divide-by-zero in PyShoe's LRT
    return max(sigma_a, 1e-5), max(sigma_w, 1e-5)

def compute_shoe_timing(imu_data: np.ndarray, sigma_a: float, sigma_w: float, W: int, G: float, has_gravity: bool = True) -> np.ndarray:
    """Computes a sample-by-sample zero-velocity boolean array.

    Uses a vectorized implementation of the SHOE likelihood ratio test to flag 
    stationary periods, optimized for high temporal resolution in timing 
    detection rather than trajectory estimation.

    Args:
        imu_data (np.ndarray): N x 6 array [acc_x, y, z, gyr_x, y, z].
        sigma_a (float): Accelerometer noise standard deviation.
        sigma_w (float): Gyroscope noise standard deviation.
        W (int): Window size in samples.
        G (float): Statistical threshold for the likelihood ratio test.
        has_gravity (bool, optional): Whether input data contains gravity.
            Defaults to True.

    Returns:
        np.ndarray: A boolean array (size N) where True indicates stationarity.
    """
    N = imu_data.shape[0]
    acc = imu_data[:, 0:3]
    gyro = imu_data[:, 3:6]
    
    inv_a = 1 / (sigma_a**2)
    inv_w = 1 / (sigma_w**2)
    
    # 1. Gyroscope Term (Rolling Sum)
    gyro_sq_mag = np.sum(gyro**2, axis=1)
    window = np.ones(W)
    gyro_term = np.convolve(gyro_sq_mag, window, mode='valid') * inv_w
    
    # 2. Acceleration Term (Rolling Sum)
    if has_gravity:
        acc_sq_mag = np.sum(acc**2, axis=1)
        sum_acc_sq = np.convolve(acc_sq_mag, window, mode='valid')
        
        sum_acc_x = np.convolve(acc[:, 0], window, mode='valid')
        sum_acc_y = np.convolve(acc[:, 1], window, mode='valid')
        sum_acc_z = np.convolve(acc[:, 2], window, mode='valid')
        sum_acc_mag = np.sqrt(sum_acc_x**2 + sum_acc_y**2 + sum_acc_z**2)
        
        g = 9.8029
        acc_term = (sum_acc_sq - 2 * g * sum_acc_mag + W * (g**2)) * inv_a
    else:
        # If the dataset is already freeAcc
        acc_sq_mag = np.sum(acc**2, axis=1)
        acc_term = np.convolve(acc_sq_mag, window, mode='valid') * inv_a
        
    # 3. Apply the threshold
    T = (acc_term + gyro_term) / W
    # print(f"DEBUG: T min: {np.min(T)}, T max: {np.max(T)}") # Add this

    # 4. Pad the tail to match the original array length
    zv = np.zeros(N, dtype=bool)
    zv[:len(T)] = T < G
    zv[len(T):] = T[-1] < G
    
    return zv

def pyshoe_process_single_file(
    segment: pd.DataFrame, 
    has_gravity: bool,
    course: str, 
    id: str, 
    clip_id: int,
    true_hs_r: pd.DataFrame | np.ndarray, 
    true_fo_r: pd.DataFrame | np.ndarray, 
    true_hs_l: pd.DataFrame | np.ndarray, 
    true_fo_l: pd.DataFrame | np.ndarray, 
    output_path: Path,
    fs: float
) -> str:
    """
    Processes a single trial dataset across all four zero-velocity update (ZUPT) 
    detectors implemented in PyShoe.

    This function extracts IMU data for both left and right feet (3-axis accelerometer 
    and 3-axis gyroscope data), applies the 'shoe', 'ared', 'amvd', and 'mbgtd' state 
    detectors, computes bilateral heel strike and foot off events, and saves separate 
    MATLAB .mat summaries for each unique detector.

    Parameters
    ----------
    segment : pd.DataFrame
        The data source for the trial. A pre-sliced Pandas DataFrame containing the target 
        time frame. Must contain a 'time' index alongside the bilateral 
        acceleration and angular velocity columns.
        
    course: str
        The name of the course or environment directory (e.g., 'course_1').
        
    id : str
        The subject identifier string matching the directory layout (e.g., 'subj_01').
        
    true_hs_r : pd.DataFrame or pd.Series or np.ndarray
        The ground truth heel strike annotations for this specific segment window.
        
    true_fo_r : pd.DataFrame or pd.Series or np.ndarray
        The ground truth foot off annotations for this specific segment window.
        
    output_path : pathlib.Path
        Path object pointing to the output directory where outputs 
        will be structured across subfolders named after each active detector.

    Returns
    ----------
    str
        A status summary string confirming completion of missing detectors, a skip message 
        if all four detectors are already found cached on disk, or an error description.
    """
    detectors_to_run = []
    for detector_name in DETECTORS:
        out_file = output_path / detector_name / f'{id}_{course}_{clip_id}.mat'
        if not out_file.exists():
            detectors_to_run.append((detector_name, out_file))
            
    if not detectors_to_run:
        return f"Skipped {id}_{course} (Already completely processed)"

    try:
        imu_left = segment[LEFT_FOOT_COLS].to_numpy() 
        imu_right = segment[RIGHT_FOOT_COLS].to_numpy() 
        time = segment["time"].to_numpy().squeeze()

        # Calculate dynamic sigmas for this specific data segment
        sigma_a_l, sigma_w_l = estimate_noise_from_midstance(imu_left, fs)
        sigma_a_r, sigma_w_r = estimate_noise_from_midstance(imu_right, fs)

        # 1. Shrink Window Size (W): Dynamically calculate a ~40ms window based on sampling frequency 
        # (e.g., 2 samples at 60Hz, 8 samples at 200Hz) to maximize temporal resolution.
        W = max(2, int(fs * 0.04))

        for detector_name, out_file in detectors_to_run:
            G_val = SPECS[detector_name]['G']

            # 2. Bypass the EKF Trajectory Math
            if detector_name == 'shoe':
                # Use your newly added ultra-fast vectorized function
                steps_left = compute_shoe_timing(imu_left, sigma_a_l, sigma_w_l, W, G_val, has_gravity)
                steps_right = compute_shoe_timing(imu_right, sigma_a_r, sigma_w_r, W, G_val, has_gravity)
            else:
                # For ARED, AMVD, MBGTD: initialize INS but ONLY compute the detector array, 
                # skipping the heavy .baseline() Kalman Filter loop completely.
                ins_l = INS(imu_left, False, sigma_a=sigma_a_l, sigma_w=sigma_w_l, T=1.0/fs)
                steps_left = ins_l.Localizer.compute_zv_lrt(W=W, G=G_val, detector=detector_name)
                
                ins_r = INS(imu_right, False, sigma_a=sigma_a_r, sigma_w=sigma_w_r, T=1.0/fs)
                steps_right = ins_r.Localizer.compute_zv_lrt(W=W, G=G_val, detector=detector_name)

            # --- Extract Timings for Left Foot ---
            padded_left = np.insert(steps_left, 0, False).astype(int)
            diff_left = np.diff(padded_left)
            
            # Initial Contact (IC) is when the stationary window begins
            HS_indices_left = np.where(diff_left == 1)[0]
            
            # 3. Correct Phase Shift: Terminal Contact (TC) is when the window ends PLUS the window length
            FO_indices_left = np.where(diff_left == -1)[0] + (W - 1)
            FO_indices_left = FO_indices_left[FO_indices_left < len(time)] # Prevent out-of-bounds

            if steps_left[0] and len(HS_indices_left) > 0:
                HS_indices_left = HS_indices_left[1:]

            HS_times_left = time[HS_indices_left]
            FO_times_left = time[FO_indices_left]

            # --- Extract Timings for Right Foot ---
            padded_right = np.insert(steps_right, 0, False).astype(int)
            diff_right = np.diff(padded_right)
            
            # Initial Contact (IC)
            HS_indices_right = np.where(diff_right == 1)[0]
            
            # 3. Correct Phase Shift: Terminal Contact (TC)
            FO_indices_right = np.where(diff_right == -1)[0] + (W - 1)
            FO_indices_right = FO_indices_right[FO_indices_right < len(time)] # Prevent out-of-bounds

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

            acc_mag_l = np.linalg.norm(imu_left[:, 0:3], axis=1)
            acc_mag_r = np.linalg.norm(imu_right[:, 0:3], axis=1)

            results = {
                'time': time,
                'acc_mag_l': acc_mag_l,
                'acc_mag_r': acc_mag_r,
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