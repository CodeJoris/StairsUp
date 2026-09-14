"""
Module: processing.tilt_correction
Description: 
    Handles the continuous 6-DOF sensor fusion of IMU data using a Madgwick 
    filter. Deconstructs the pipeline into single-task functions for unit 
    normalization, orientation estimation, frame rotation, and gravity removal.

Dependencies:
    - numpy
    - pandas
    - scipy.spatial.transform
    - ahrs
"""

import numpy as np
import pandas as pd
from typing import List
from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation



def standardize_acceleration_units(
    accel_data: np.ndarray, 
    unit: str
) -> np.ndarray:
    """
    Standardizes accelerometer data to units of gravitational acceleration (g).

    Parameters
    ----------
    accel_data : np.ndarray
        An Nx3 array of raw accelerometer data.
    unit : str
        The unit of the input data. Must be either 'g' or 'm/s^2'.

    Returns
    -------
    np.ndarray
        An Nx3 array of accelerometer data expressed in g's.

    Raises
    ------
    ValueError
        If an unsupported unit is provided.
    """
    if unit == 'g':
        return accel_data.copy()
    elif unit == 'm/s^2':
        return accel_data / 9.81
    else:
        raise ValueError("Unit must be either 'g' or 'm/s^2'.")


def compute_madgwick_quaternions(
    accel_data: np.ndarray, 
    gyro_data: np.ndarray, 
    sample_rate: float, 
    beta: float
) -> np.ndarray:
    """
    Computes spatial orientation as quaternions using a Madgwick filter.

    Parameters
    ----------
    accel_data : np.ndarray
        An Nx3 array of normalized accelerometer data.
    gyro_data : np.ndarray
        An Nx3 array of gyroscope data in rad/s.
    sample_rate : float
        The sampling frequency in Hz.
    beta : float
        The Madgwick filter gain.

    Returns
    -------
    np.ndarray
        An Nx4 array of quaternions in scalar-first format [w, x, y, z].
    """
    madgwick_filter = Madgwick(
        gyr=gyro_data, 
        acc=accel_data, 
        frequency=sample_rate, 
        gain=beta
    )
    return madgwick_filter.Q


def rotate_to_global_frame(
    accel_data: np.ndarray, 
    quaternions: np.ndarray
) -> np.ndarray:
    """
    Rotates body-frame acceleration vectors into the global coordinate frame.

    Parameters
    ----------
    accel_data : np.ndarray
        An Nx3 array of body-frame acceleration data.
    quaternions : np.ndarray
        An Nx4 array of quaternions in [w, x, y, z] format.

    Returns
    -------
    np.ndarray
        An Nx3 array of global-frame acceleration data.
    """
    # Convert AHRS quaternions [w, x, y, z] to SciPy format [x, y, z, w]
    scipy_quats = np.column_stack((quaternions[:, 1:], quaternions[:, 0]))
    
    # Generate rotation matrices and apply to acceleration data
    rotations = Rotation.from_quat(scipy_quats)
    return rotations.apply(accel_data)


def remove_static_gravity(
    global_accel_data: np.ndarray, 
    unit: str
) -> np.ndarray:
    """
    Subtracts the static gravity vector from the global vertical axis (Z).

    Parameters
    ----------
    global_accel_data : np.ndarray
        An Nx3 array of global-frame acceleration data.
    unit : str
        The original unit of the data ('g' or 'm/s^2') to determine the 
        magnitude of the gravity vector to subtract.

    Returns
    -------
    np.ndarray
        An Nx3 array of purely dynamic global acceleration data.
    """
    gravity_magnitude = 1.0 if unit == 'g' else 9.81
    gravity_vector = np.array([0.0, 0.0, gravity_magnitude])
    
    return global_accel_data - gravity_vector


def align_and_correct_tilt(
    continuous_df: pd.DataFrame,
    accel_cols: List[str],
    gyro_cols: List[str],
    sample_rate: float,
    beta: float = 0.1,
    unit: str = 'g',
    quat_cols: List[str] = None
) -> pd.DataFrame:
    """
    Orchestrates the tilt correction and gravity removal pipeline.

    This function extracts sensor data, delegates to single-task 
    processing functions, and appends the corrected orientations 
    and dynamic accelerations to a new dataframe.

    Parameters
    ----------
    continuous_df : pd.DataFrame
        The dataframe containing the synchronized IMU kinematic data.
    accel_cols : List[str]
        A list of exactly three string column names (X, Y, Z) for acceleration.
    gyro_cols : List[str]
        A list of exactly three string column names (X, Y, Z) for gyroscope.
    sample_rate : float
        The static sampling frequency in Hz.
    beta : float, optional
        The Madgwick filter gain. Defaults to 0.1.
    unit : str, optional
        The unit of the accelerometer data ('g' or 'm/s^2'). Defaults to 'g'.
    quat_cols : List[str], optional
        A list of exactly four string column names (qw, qx, qy, qz) for quaternions.

    Returns
    -------
    pd.DataFrame
        A deep copy of the original dataframe with appended columns for 
        quaternions and global dynamic accelerations.

    Raises
    ------
    ValueError
        If column lists are not length 3 or if columns are missing.
    """
    # Validate inputs
    if gyro_cols is None and quat_cols is None:
        raise ValueError("Either gyro_cols or quat_cols must be provided.")

    if len(accel_cols) != 3 or (gyro_cols is not None and len(gyro_cols) != 3):
        raise ValueError("accel_cols and gyro_cols must contain exactly 3 column names.")
    
    # Create a deep copy to prevent SettingWithCopyWarnings[cite: 2]
    df_corrected = continuous_df.copy()

    # Extract arrays
    accel_data = continuous_df[accel_cols].to_numpy()
    gyro_data = continuous_df[gyro_cols].to_numpy() if gyro_cols is not None else None
    accel_norm = standardize_acceleration_units(accel_data, unit)
    if quat_cols is not None:
        quaternions = continuous_df[quat_cols].to_numpy()
    else:
        quaternions = compute_madgwick_quaternions(accel_norm, gyro_data, sample_rate, beta)

    # We rotate the original acceleration data to preserve the original units in the output
    accel_global = rotate_to_global_frame(accel_data, quaternions)
    dynamic_accel_global = remove_static_gravity(accel_global, unit)

    # Append results
    df_corrected['q_w'] = quaternions[:, 0]
    df_corrected['q_x'] = quaternions[:, 1]
    df_corrected['q_y'] = quaternions[:, 2]
    df_corrected['q_z'] = quaternions[:, 3]

    df_corrected[f"{accel_cols[0]}_corrected"] = dynamic_accel_global[:, 0]
    df_corrected[f"{accel_cols[1]}_corrected"] = dynamic_accel_global[:, 1]
    df_corrected[f"{accel_cols[2]}_corrected"] = dynamic_accel_global[:, 2]

    return df_corrected

if __name__ == "__main__":
    from pathlib import Path
    from gaitmap_datasets import StairAmbulationHealthy2021PerTest

    dataset_base_path = Path(__file__).parent.parent.parent.parent / "data" / "osf"

    dataset = StairAmbulationHealthy2021PerTest(
        data_folder=Path(dataset_base_path), 
        include_pressure_data=True,
        include_hip_sensor = True
    )

    ACCEL_COLS = ['acc_x', 'acc_y', 'acc_z']
    GYRO_COLS = ['gyr_x', 'gyr_y', 'gyr_z']

    for datapoint in dataset:
        df = datapoint.data
        fs = datapoint.sampling_rate_hz

        hip_data = df['hip_sensor']
        # convert deg/s to rad/s
        hip_data[GYRO_COLS] = np.deg2rad(hip_data[GYRO_COLS])

        df_corrected = align_and_correct_tilt(
            continuous_df=hip_data,
            accel_cols=ACCEL_COLS,
            gyro_cols=GYRO_COLS,
            sample_rate=fs,
            beta=0.1,
            unit='m/s^2',
        )

        x_mean = df_corrected[ACCEL_COLS[0]].mean()
        y_mean = df_corrected[ACCEL_COLS[1]].mean()
        z_mean = df_corrected[ACCEL_COLS[2]].mean()

        old_x_mean = hip_data[ACCEL_COLS[0]].mean()
        old_y_mean = hip_data[ACCEL_COLS[1]].mean()
        old_z_mean = hip_data[ACCEL_COLS[2]].mean()

        print(f'old averages: {old_x_mean}, {old_y_mean}, {old_z_mean}')

        print(f'new averages: {x_mean}, {y_mean}, {z_mean}')


        break

        df_corrected = align_and_correct_tilt(
            continuous_df=df,
            free_acc = True,
            accel_cols=ACCEL_COLS,
            gyro_cols=GYRO_COLS,
            sample_rate=fs,
            beta=0.1,
            unit='g',
        )
