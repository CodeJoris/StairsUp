'''
CWT (Continuous Wavelet Transform) functions for foot strike detection.
'''  
from scipy.signal import find_peaks
import numpy as np
import pywt  
import pandas as pd
from scipy.io import savemat

COLS = ['acceleration_Pelvis_x', 'acceleration_Pelvis_y', 'acceleration_Pelvis_z']
SAMPLING_FREQUENCY = 60

def compute_cwt(signal, fs, step_freq, wavelet='mexh'):
    '''
    Computes the Continuous Wavelet Transform (CWT) of a given signal 
    given a sampling frequency, step frequency, and wavelet.

    Parameters
    ----------
        signal (array-like): The input signal to be transformed.
        fs (float): The sampling frequency of the signal.
        step_freq (float): The physical step frequency expected in the signal.
        wavelet (str): The type of wavelet to use for the CWT. Default is 'mexh' (Mexican Hat).

    Returns
    -------
        coefs (ndarray): The CWT coefficients of the input signal.
        scale (float): The scale used for the CWT based on the step frequency.
    '''
    fc = pywt.central_frequency(wavelet)
    scale = (fc * fs) / step_freq
    coefs, _ = pywt.cwt(signal, scales=[scale], wavelet=wavelet)
    return coefs[0], scale

def predict_step_timing_from_cwt(cwt_output, time_array, min_distance=None):
    '''
    Predicts the timing of step events from the CWT output.

    Parameters
    ----------
        cwt_output (array-like): The CWT coefficients of the input signal.
        time_array (array-like): The time array corresponding to the CWT output in ms.
        min_distance (float, optional): The minimum distance between peaks. Default is None.

    Returns
    -------
        hs_times (ndarray): The times of the detected peaks in ms.
    '''
    peaks, _ = find_peaks(cwt_output, distance=min_distance)
    return peaks, time_array[peaks]

def predict_foot_off_from_secondary_cwt(cwt1_output, time_array, hs_peak_indices, fs, step_freq):
    '''
    Predicts foot off events from a secondary CWT output based on the detected heel strike peaks.
    
    Parameters
    ----------
        cwt1_output (array-like): The CWT coefficients of the input signal.
        time_array (array-like): The time array corresponding to the CWT output in ms.
        hs_peak_indices (array-like): The indices of the detected heel strike peaks.
        fs (float): The sampling frequency of the signal.
        step_freq (float): The physical step frequency expected in the signal.

    Returns
    -------
        fo_times (ndarray): The times of the detected foot off events in ms.
    '''
    cwt2_output, _ = compute_cwt(cwt1_output, fs, step_freq=step_freq, wavelet='mexh')
    fo_indices = []
    for i in range(len(hs_peak_indices) - 1):
        idx_start = hs_peak_indices[i]
        idx_end = hs_peak_indices[i + 1]
        if idx_end > idx_start + 1:
            segment_min_idx = idx_start + np.argmin(cwt2_output[idx_start:idx_end])
            fo_indices.append(segment_min_idx)
    fo_indices = np.array(fo_indices, dtype=int)
    if len(fo_indices) == 0:
        return np.array([])
    return time_array[fo_indices]

def cwt_process_single_file(
    file_path, 
    course, 
    id, 
    y_HS_true, 
    y_FO_true, 
    data_path, 
    fs,
    step_freq=1.5,
    cols=COLS,
    vertical_index=2,
    component='z'):
    '''
    Processes a single file for foot strike detection using Continuous Wavelet Transform (CWT).
    
    Parameters
    ----------
        file_path (str): Path to the input CSV file containing acceleration data.
        course (str): The course identifier for the data.
        id (str): The unique identifier for the subject or trial.
        y_HS_true (ndarray): The true heel strike times.
        y_FO_true (ndarray): The true foot off times.
        data_path (Path): The base path for saving results.
        step_freq (float, optional): The expected physical step frequency in Hz. Default is 1.5 Hz.
        cols (list, optional): The columns to use from the CSV file. Default is COLS.
        vertical_index (int, optional): The index of the vertical acceleration component in cols. Default is 2.
        component (str, optional): The acceleration component to use ('z,y,z' or 'mag'). Default is 'z'.
    Returns
    -------
        str: A message indicating the success or failure of the processing.

    Raises
    ------
        ValueError: If the specified component is not 'z' or 'mag'.
    '''    
    output_dir = data_path / "toolbox_results" / f"cwt_{step_freq}_{component}"
    output_file = output_dir / f"{id}_{course}.mat"

    if output_file.exists():
        return f"Skipped {id}_{course}.mat (already processed)"

    try:
        acceleration_data = pd.read_csv(file_path, usecols=cols + ['time']) # time col is in ms

        if component == 'mag':
            accel_array = np.linalg.norm(acceleration_data[cols].to_numpy(), axis=1)
        else:
            accel_array = acceleration_data[cols[vertical_index]].to_numpy()

        coefs, scale = compute_cwt(accel_array, fs=fs, step_freq=step_freq, wavelet='mexh')

        # Retrieve both indices and times
        HS_indices, HS_times = predict_step_timing_from_cwt(
            cwt_output=coefs, 
            time_array=acceleration_data['time'].to_numpy()
        )
        
        # Run secondary CWT using raw arrays and indices
        FO_times = predict_foot_off_from_secondary_cwt(
            cwt1_output=coefs, 
            time_array=acceleration_data['time'].to_numpy(),
            hs_peak_indices=HS_indices, 
            fs=fs,
            step_freq=step_freq
        )

        y_HS_pred = np.zeros((len(HS_times), 2)) # expected structure by the rest of the pipeline
        y_HS_pred[:,0] = HS_times

        y_FO_pred = np.zeros((len(FO_times), 2)) # expected structure by the rest of the pipeline
        y_FO_pred[:,0] = FO_times

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

        return f'{id}_{course} Success'
    
    except Exception as e:
        return f"FAILED: {id}_{course} - Error: {str(e)}"