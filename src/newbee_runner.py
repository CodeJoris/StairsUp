"""
Module: newbee_runner.py
Description: 
    Serves as the main entry point for processing the NewBee dataset. 
    Extracts target sequences, prepares the data, and processes it using 
    the KielMAT, and PyShoe toolboxes. Supports parallel processing.

    Newbee units: m/s^2 rads/s

Dependencies:
    - pandas
    - numpy
    - pathlib
    - concurrent.futures
"""
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from processing.extract_golden_standard import extract_golden_standard
from Toolboxes.KielMAT.kielmat_export import kielmat_process_single_file
from Toolboxes.PyShoe.pyshoe_export import pyshoe_process_single_file
from processing.newbee_surface_extraction import extract_surface_segments

SAMPLING_FREQUENCY = 60
DATA_PATH = Path(__file__).resolve().parent.parent / "data"
TARGET_MODE = "stairs_up" 
OUTPUT_PATH = DATA_PATH / f'newbee_{TARGET_MODE}'
COLS = ["time", "insoles_RightFoot_is_step", "insoles_LeftFoot_is_step", "insoles_RightFoot_is_lifted", "insoles_LeftFoot_is_lifted"]

def get_ytrue_for_stairs(course: str, id: str) -> list:
    """
    Extracts the 'golden standard' heel strike and toe off annotations.

    This function reads the target trial's 'labels.csv' file, identifies continuous 
    time windows spent walking up stairs using `get_stair_segments`, parses the step 
    annotation columns, and slices them into isolated dictionary packages.

    Parameters
    ----------
    course : str
        The name of the course or environment directory (e.g., 'courseA').
    id : str
        The subject identifier string matching the directory structure (e.g., 'id01').

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents an isolated continuous
        staircase climb segment. Returns an empty list if no segments are found.

    Raises
    ------
    Exception
        Catches and prints underlying parsing errors (e.g., missing columns, invalid paths).
    """
    try:
        file_path = DATA_PATH / "newbee" / course / id / "labels.csv"

        stair_blocks = extract_surface_segments(str(file_path), target_mode=TARGET_MODE)
        if stair_blocks.empty:
            return [] 

        data = pd.read_csv(file_path, usecols=COLS)
        full_golden_standard = extract_golden_standard(data) 
        
        y_HS_full, y_FO_full = full_golden_standard

        staircase_clips_ground_truth = []
        for idx, block in stair_blocks.iterrows():
            mask_HS = (y_HS_full[:, 0] >= block.start_time) & (y_HS_full[:, 0] <= block.end_time)
            mask_FO = (y_FO_full[:, 0] >= block.start_time) & (y_FO_full[:, 0] <= block.end_time)

            hs_window = y_HS_full[mask_HS]
            fo_window = y_FO_full[mask_FO]

            def split_feet(arr):
                if len(arr) == 0: return np.array([]), np.array([])
                return arr[arr[:, 1] == 0][:, 0], arr[arr[:, 1] == 1][:, 0]

            true_hs_r, true_hs_l = split_feet(hs_window)
            true_fo_r, true_fo_l = split_feet(fo_window)

            def clean_boundaries(hs, fo):
                if len(fo) > 0 and len(hs) > 0 and fo[0] < hs[0]:
                    fo = fo[1:]
                if len(hs) > 0 and len(fo) > 0 and hs[-1] > fo[-1]:
                    hs = hs[:-1]
                return hs, fo

            true_hs_r, true_fo_r = clean_boundaries(true_hs_r, true_fo_r)
            true_hs_l, true_fo_l = clean_boundaries(true_hs_l, true_fo_l)
        
            if len(true_hs_r) == 0 and len(true_hs_l) == 0:
                continue

            staircase_clips_ground_truth.append({
                'segment_id': idx,
                'start_time': block.start_time,
                'end_time': block.end_time,
                'y_HS_r': true_hs_r,
                'y_FO_r': true_fo_r,
                'y_HS_l': true_hs_l,
                'y_FO_l': true_fo_l
            })

        return staircase_clips_ground_truth
    
    except Exception as e:
        print(f"ERR loading ground truth for {id}_{course}: {str(e)}")
        return None
    
def process_file_all_toolboxes(file_path: Path) -> str:
    """
    Runs gait analysis toolboxes across isolated 'stairs_up' clips.

    Extracts subject and course metadata dynamically from the provided 
    file path, extracts the timestamp-segmented ground truth, and sequentially evaluates 
    the data through KielMAT, PyShoe.

    Parameters
    ----------
    file_path : pathlib.Path
        The absolute Path object pointing to the high-frequency raw sensor data file.

    Returns
    -------
    str
        A formatted status summary string detailing the processing outcome.
    """
    id = file_path.parts[-2]
    course = file_path.parts[-3]

    stair_clips_gt = get_ytrue_for_stairs(course, id)
    if not stair_clips_gt:
        return f'SKIPPED: {id}_{course} - No stairs_up data found or load failed.'
    
    raw_sensor_df = pd.read_csv(file_path)
    results_summary = []
    
    for clip in stair_clips_gt:
        sensor_clip = raw_sensor_df[(raw_sensor_df.time >= clip['start_time']) & 
                                    (raw_sensor_df.time <= clip['end_time'])]
        
        res_kielmat = kielmat_process_single_file(
            sensor_clip, course, id, clip['segment_id'], 
            clip['y_HS_r'], clip['y_FO_r'], 
            clip['y_HS_l'], clip['y_FO_l'], 
            OUTPUT_PATH, SAMPLING_FREQUENCY
        )

        min_spacings = [600.0]  # Example values in milliseconds
        to_fo_delays = [250.0]
        num_of_stds = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]

        for min_spacing in min_spacings:
            for to_fo_delay in to_fo_delays:
                for n_of_std in num_of_stds:
                    res_pyshoe = pyshoe_process_single_file(
                        sensor_clip, course, id, clip['segment_id'], 
                        clip['y_HS_r'], clip['y_FO_r'], 
                        clip['y_HS_l'], clip['y_FO_l'], 
                        OUTPUT_PATH, SAMPLING_FREQUENCY,
                        has_gravity=False,
                        min_spacing_ms=min_spacing,
                        to_fo_delay_ms=to_fo_delay,
                        n_of_std=n_of_std
                    )
                
        results_summary.append(
            f"Clip {clip['segment_id']}: KielMAT [{res_kielmat}] | PyShoe [{res_pyshoe}]"
        )
        
    return f"{id}_{course} => " + " | ".join(results_summary)

if __name__ == "__main__":
    file_list = list(DATA_PATH.rglob("xsens.csv"))
    print(f"Found {len(file_list)} files to process.")

    if not file_list:
        print("No files found. Exiting.")
        exit()

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file_all_toolboxes, file_list)

        for r in results:
            print(r)