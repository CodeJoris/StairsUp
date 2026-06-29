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
    y_HS_true: pd.DataFrame | np.ndarray, 
    y_FO_true: pd.DataFrame | np.ndarray, 
    data_path: Path
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
        
    data_path : pathlib.Path
        The base Path object pointing to the global data directory where outputs 
        will be structured under 'toolbox1/KielMAT/'.

    Returns:
    --------
    str
        A status string indicating processing success, a skipped status message 
        if the destination file already exists on disk, or a detailed error failure trace.
    """
    output_dir = data_path / "toolbox1" / "KielMAT"
    output_file = output_dir / f"{id}_{course}_{clip_id}.mat"

    if output_file.exists():
        return f"Skipped {id}_{course}_clip_{clip_id}.mat (already processed)"

    try:
        acceleration_data = segment[COLS] # KielMAT takes in a pandas dataframe

        vertical_accel_array = acceleration_data['acceleration_Pelvis_z'].to_numpy()

        HS_times, FO_times = signal_decomposition_algorithm(
            vertical_accelerarion_data=vertical_accel_array,
            initial_sampling_frequency=SAMPLING_FREQUENCY
        )

        temp = HS_times*1000 # heel strike times in ms
        y_HS_pred = np.zeros((len(temp), 2)) # expected structure by the rest of the pipeline
        y_HS_pred[:,0] = temp 

        temp = FO_times*1000 # foot off times in ms
        y_FO_pred = np.zeros((len(temp), 2)) # expected structure by the rest of the pipeline
        y_FO_pred[:,0] = temp 

        # Create a structure matching what the pipeline expects:
        # results.y and results.y_hat
        results = {
            'y_HS': y_HS_true,         # true
            'y_FO': y_FO_true,
            'y_hat_HS': y_HS_pred,     # pred
            'y_hat_FO': y_FO_pred
        }

        output_dir.mkdir(parents = True, exist_ok = True)
        savemat(output_file, {'results': results})

        return f'{id}_{course}_{clip_id} Success'
    
    except Exception as e:
        return f"FAILED: {id}_{course}_{clip_id} - Error: {str(e)}"