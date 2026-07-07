import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import savemat

from kielmat.utils.preprocessing import signal_decomposition_algorithm

# Define global constants
COLS = ['acceleration_Pelvis_x', 'acceleration_Pelvis_y', 'acceleration_Pelvis_z']
SAMPLING_FREQUENCY = 60



def kielmat_process_single_file(
    segment: pd.DataFrame, 
    course: str, 
    id: str, 
    clip_id: int,  # Added to separate staircase events
    true_hs_r: pd.DataFrame | np.ndarray, 
    true_fo_r: pd.DataFrame | np.ndarray, 
    true_hs_l: pd.DataFrame | np.ndarray, 
    true_fo_l: pd.DataFrame | np.ndarray, 
    output_path: Path
) -> str:
    """
    Processes a single trial dataset using KielMAT's signal decomposition algorithm 
    to predict heel strike (HS) and foot off (FO) event timestamps.

    This function isolates the vertical acceleration channel from the pelvis IMU sensor, 
    identifies temporal gait events, structures both predictions and ground truth annotations 
    into a structured dictionary, and exports the output into a MATLAB .mat format.

    Parameters:
    -----------
    segment : pd.DataFrame
        The data source for the trial. A pre-sliced Pandas DataFrame containing the target 
        time frame. Must contain at minimum the column: 'acceleration_Pelvis_z'.
        
    course : str
        The name of the course or environment directory (e.g., 'courseA').
        
    id : str
        The subject identifier string matching the directory layout (e.g., 'id01').
    
    clip_ip : int
        The segment clip id in order to handle multiple stairs section in one run.
        
    y_HS_true : pd.DataFrame or pd.Series or np.ndarray
        The ground truth heel strike annotations for this specific segment window.
        
    y_FO_true : pd.DataFrame or pd.Series or np.ndarray
        The ground truth foot off annotations for this specific segment window.
        
    ouput_path : pathlib.Path
        The base Path object pointing to the output directory where outputs 
        will be structured under 'output_path/KielMAT/'.

    Returns:
    --------
    str
        A status string indicating processing success, a skipped status message 
        if the destination file already exists on disk, or a detailed error failure trace.
    """
    output_dir = output_path / "KielMAT"
    output_file = output_dir / f"{id}_{course}_{clip_id}.mat"

    if output_file.exists():
        return f"Skipped {id}_{course}_clip_{clip_id}.mat (already processed)"

    try:
        acceleration_data = segment[COLS] # KielMAT takes in a pandas dataframe
        time = segment['time'].to_numpy().squeeze()

        vertical_accel_array = acceleration_data['acceleration_Pelvis_z'].to_numpy()

        HS_times, FO_times = signal_decomposition_algorithm(
            vertical_accelerarion_data=vertical_accel_array,
            initial_sampling_frequency=SAMPLING_FREQUENCY
        )

        temp = (HS_times*1000) + time[0] # heel strike times in ms
        y_HS_pred = np.zeros((len(temp), 2)) # expected structure by the rest of the pipeline
        y_HS_pred[:,0] = temp 

        temp = (FO_times*1000) + time[0] # foot off times in ms
        y_FO_pred = np.zeros((len(temp), 2)) # expected structure by the rest of the pipeline
        y_FO_pred[:,0] = temp 

        def tag_foot(times, label):
            if len(times) == 0: return np.empty((0, 2))
            return np.column_stack((times, np.full(len(times), label)))


        true_hs_r_tagged = tag_foot(true_hs_r, 0)
        true_hs_l_tagged = tag_foot(true_hs_l, 0)
        true_fo_r_tagged = tag_foot(true_fo_r, 0)
        true_fo_l_tagged = tag_foot(true_fo_l, 0)

        y_HS_true = np.vstack((true_hs_r_tagged, true_hs_l_tagged)) if len(true_hs_r_tagged) or len(true_hs_l_tagged) else np.empty((0, 2))
        y_FO_true = np.vstack((true_fo_r_tagged, true_fo_l_tagged)) if len(true_fo_r_tagged) or len(true_fo_l_tagged) else np.empty((0, 2))

        if len(y_HS_true): y_HS_true = y_HS_true[y_HS_true[:, 0].argsort()]
        if len(y_FO_true): y_FO_true = y_FO_true[y_FO_true[:, 0].argsort()]

        acc_mag = np.linalg.norm(acceleration_data.values, axis=1)

        # Create a structure matching what the pipeline expects:
        # results.y and results.y_hat
        results = {
            "time": time,
            'y_HS': y_HS_true,         # true
            'acc_mag_l': acc_mag,
            'y_FO': y_FO_true,
            'y_hat_HS': y_HS_pred,     # pred
            'y_hat_FO': y_FO_pred
        }

        output_dir.mkdir(parents = True, exist_ok = True)
        savemat(output_file, {'results': results})

        return f'{id}_{course}_{clip_id} Success'
    
    except Exception as e:
        return f"FAILED: {id}_{course}_{clip_id} - Error: {str(e)}"