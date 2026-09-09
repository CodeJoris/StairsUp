"""
Module: test_skdh_lumbar.py
Description: 
    Handles the processing of continuous dataset segments using the SKDH 
    GaitLumbar algorithm and exports the predicted gait events to MATLAB 
    format for standardized evaluation.

Dependencies:
    - numpy
    - pandas
    - pathlib
    - scipy.io
    - skdh
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import savemat
import skdh
import warnings
import matplotlib.pyplot as plt
from tests.extract_golden_standard import extract_golden_standard


accel_cols = ['acceleration_Pelvis_x', 'acceleration_Pelvis_y', 'acceleration_Pelvis_z']
xsens_file = Path(__file__).resolve().parent.parent.parent / "data" / "newbee" / "courseA" / "id01" / "xsens.csv"
labels_file = Path(__file__).resolve().parent.parent.parent / "data" / "newbee" / "courseA" / "id01" / "labels.csv"

def skdh_process_single_file(file):
    segment = pd.read_csv(file)

    try:
        time_ms = segment['time'].to_numpy().squeeze()
        time_sec = time_ms / 1000.0 
        fs = 1 / np.mean(np.diff(time_sec)) 
        
        # Scale m/s^2 to standard G for SKDH thresholds
        accel_data = segment[accel_cols].to_numpy() / 9.80665
        acc_mag = np.linalg.norm(accel_data, axis=1)

        # Run the GaitLumbar algorithm silently to bypass pendulum errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gait = skdh.gait.GaitLumbar()
            skdh_res = gait.predict(time=time_sec, accel=accel_data, fs=fs)
        
        ic_indices = skdh_res.get('IC', [])
        fc_indices = skdh_res.get('FC', [])
        
        HS_times = time_ms[ic_indices] if len(ic_indices) > 0 else np.array([])
        FO_times = time_ms[fc_indices] if len(fc_indices) > 0 else np.array([])

        return acc_mag, HS_times, FO_times, ic_indices, fc_indices
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        return None, None, None


# plot the acceleration detected HS and true HS
y_HS, y_FO = extract_golden_standard(pd.read_csv(labels_file))
acc_mag, HS_times, FO_times, ic_indices, fc_indices = skdh_process_single_file(xsens_file)

print(HS_times, FO_times)


# print(type(acc_mag), type(HS_times), type(FO_times))
if False:
    y_HS_indices, y_FO_indices = y_HS[:, 0].astype(int), y_FO[:, 0].astype(int)
    acc_min, acc_max = np.min(acc_mag), np.max(acc_mag)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc_mag)
    plt.scatter(HS_times, acc_mag[ic_indices], color='red', label='Detected HS')
    for y_hs in y_HS_indices:
        plt.vlines(y_hs, acc_min, acc_max, color='blue', label='True HS')
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (G)')
    plt.title('Gait Events - Detected vs True')

    plt.xlim(0, 20000)


    plt.subplot(1, 2, 2)
    plt.plot(acc_mag)
    plt.scatter(FO_times, acc_mag[fc_indices], color='red', label='Detected FO')
    for y_fo in y_FO_indices:
        plt.vlines(y_fo, acc_min, acc_max, color='blue', label='True FO')
    plt.xlabel('Time (ms)')
    plt.ylabel('Acceleration (G)')
    plt.title('Gait Events - Detected vs True')

    plt.xlim(0, 20000)

    plt.show()



