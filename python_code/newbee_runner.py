import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Import your preprocessing and toolbox functions
from python_code.src.preprocessing.extract_golden_standard import extract_golden_standard
from python_code.Toolboxes.KielMAT.kielmat_export import kielmat_process_single_file
from python_code.Toolboxes.PyShoe.pyshoe_export import pyshoe_process_single_file
from python_code.src.preprocessing.extract_stairs import get_stair_segments

# Constants
SAMPLING_FREQUENCY = 60
DATA_PATH = Path(__file__).resolve().parent.parent / "data"
TARGET_MODE = "stairs_up" # CHANGE THIS: to see other surfaces
OUTPUT_PATH = DATA_PATH / TARGET_MODE
COLS = ["time", "insoles_RightFoot_is_step", "insoles_LeftFoot_is_step", "insoles_RightFoot_is_lifted", "insoles_LeftFoot_is_lifted"]

def get_ytrue_for_stairs(course: str, id: str) -> list:
    """
    Extracts the 'golden standard' heel strike and toe off annotations for a specific 
    subject and course, filtered down and split into individual 'stairs_up' segments.

    This function reads the target trial's 'labels.csv' file, identifies the continuous 
    time windows spent walking up stairs using `get_stair_segments`, parses the step 
    annotation columns, and slices them into isolated dictionary packages.

    Args:
        course: (str) The name of the course or environment directory (e.g., 'courseA').
        id: (str) The subject identifier string matching the directory structure (e.g., 'id01').

    Returns:
        list of dict\n
            A list of dictionaries, where each dictionary represents an isolated continuous\n
            staircase climb segment. Returns an empty list [] if no 'stairs_up' segments\n
            are present in the trial. Each dictionary contains:\n
                - `segment_id` (int): The sequential index of the staircase climb.\n
                - `start_time` (float/int): The starting timestamp (ms) of the climb.\n
                - `end_time` (float/int): The ending timestamp (ms) of the climb.\n
                - `y_HS` (pd.DataFrame/Series): Sliced heel strike ground truth matching the window.\n
                - `y_FO` (pd.DataFrame/Series): Sliced toe off ground truth matching the window.\n

    Raises:
        Exception
            Catches and prints any underlying parsing errors (e.g., missing columns, 
            invalid file paths) and returns None.
    """
    try:
        file_path = DATA_PATH / "data_set" / course / id / "labels.csv"

        # 1. Get our stairs segments (start and end times)
        stair_blocks = get_stair_segments(str(file_path), target_mode=TARGET_MODE)
        if stair_blocks.empty:
            return []  # No stairs up in this entire trial

        # 2. Ingest the full golden standard columns from labels.csv
        data = pd.read_csv(file_path, usecols=COLS)
        full_golden_standard = extract_golden_standard(data) # returns 1D numpy arrays with timestamps
        
        y_HS_full, y_FO_full = full_golden_standard

        # 3. Filter down to ONLY the segments where stairs up occurs
        staircase_clips_ground_truth = []
        for idx, block in stair_blocks.iterrows():
        # Mask using the first column [:, 0] for time evaluation
            mask_HS = (y_HS_full[:, 0] >= block.start_time) & (y_HS_full[:, 0] <= block.end_time)
            mask_FO = (y_FO_full[:, 0] >= block.start_time) & (y_FO_full[:, 0] <= block.end_time)

            hs_window = y_HS_full[mask_HS]
            fo_window = y_FO_full[mask_FO]

            # Helper to separate the Nx2 array into Right (0) and Left (1)
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
        
            # Safety Guard: Skip if NO steps exist on either foot
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
    Worker function executed by ProcessPoolExecutor to run gait analysis toolboxes 
    across all isolated 'stairs_up' clips found within a specific raw trial file.

    It extracts the subject and course metadata dynamically from the provided 
    file path, extracts the timestamp-segmented ground truth, and sequentially evaluates 
    the data through both the KielMAT and PyShoe toolboxes for every individual 
    staircase climb detected.

    Parameters:
    -----------
    file_path : pathlib.Path
        The absolute Path object pointing to the high-frequency raw sensor data file 
        (e.g., an 'xsens.csv' file). The file path structure must match 
        '.../[course]/[id]/xsens.csv' to allow correct metadata parsing.

    Returns:
    --------
    str
        A formatted status summary string detailing the processing outcome. 
        - If successful: Returns a combined log of metrics for all processed staircase clips.
        - If skipped or failed: Returns an error/skip string flagging why processing halted.
    """
    id = file_path.parts[-2]
    course = file_path.parts[-3]

    # Get segmented ground truth data for each staircase climb
    stair_clips_gt = get_ytrue_for_stairs(course, id)
    if not stair_clips_gt:
        return f'SKIPPED: {id}_{course} - No stairs_up data found or load failed.'
    
    raw_sensor_df = pd.read_csv(file_path)

    results_summary = []
    
    # Process each isolated staircase section through your toolboxes
    for clip in stair_clips_gt:
        # Note: You will need your toolboxes to accept time constraints 
        # or pass the pre-sliced `file_path` clips you exported earlier!
        sensor_clip = raw_sensor_df[(raw_sensor_df.time >= clip['start_time']) & 
                                    (raw_sensor_df.time <= clip['end_time'])]
        # res_kielmat = kielmat_process_single_file(sensor_clip, course, id, clip['segment_id'], clip['y_HS'], clip['y_FO'], DATA_PATH)
        res_kielmat = "Not on"
        res_pyshoe = pyshoe_process_single_file(
            sensor_clip, False, course, id, clip['segment_id'], 
            clip['y_HS_r'], clip['y_FO_r'], 
            clip['y_HS_l'], clip['y_FO_l'], 
            OUTPUT_PATH, SAMPLING_FREQUENCY
        )
        
        # results_summary.append(f"Clip {clip['segment_id']}: KielMAT [{res_kielmat}] | PyShoe [{res_pyshoe}]")
        results_summary.append(f"Clip {clip['segment_id']}: PyShoe [{res_pyshoe}]")
        
    return f"{id}_{course} => " + " | ".join(results_summary)

if __name__ == "__main__":
    file_list = list(DATA_PATH.rglob("xsens.csv"))
    print(f"Found {len(file_list)} files to process.")

    if not file_list:
        print("No files found. Exiting.")
        exit()

    # Multiprocessing
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_file_all_toolboxes, file_list)

        for r in results:
            print(r)
