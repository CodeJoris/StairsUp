import os
from pathlib import Path
import numpy as np
import pandas as pd

def get_stair_segments(f_label: str, target_mode: str = "stairs_up") -> pd.DataFrame:
    """
    Parses a labels.csv file to extract continuous blocks of a specific walking mode.
    
    Parameters:
    -----------
    f_label : str
        The full path to the labels.csv file.
    target_mode : str
        The walking mode string to isolate (e.g., 'stairs_up', 'stairs_down').
        
    Returns:
    --------
    pd.DataFrame
        A DataFrame with 'start_time' and 'end_time' columns for each continuous segment.
    """
    if not os.path.exists(f_label):
        raise FileNotFoundError(f"Label file not found at: {f_label}")
        
    # 1. Load only the essential columns to preserve memory
    df = pd.read_csv(f_label, usecols=['time', 'walk_mode'])
    
    # 2. Identify rows matching the target activity
    is_target = df['walk_mode'] == target_mode
    
    # If the mode doesn't exist in this file, return an empty tracking DataFrame
    if not is_target.any():
        return pd.DataFrame(columns=['start_time', 'end_time'])
        
    # 3. Detect changes in the activity state to find block boundaries
    # A true/false shift flags exactly when someone starts or stops climbing stairs
    df['segment_id'] = (is_target != is_target.shift()).cumsum()
    
    # 4. Filter out everything else, leaving only target rows grouped by segment chunks
    target_df = df[is_target]
    
    # 5. Extract the boundary timestamps for each individual continuous run
    segments = (
        target_df.groupby('segment_id')['time']
        .agg(['min', 'max'])
        .reset_index(drop=True)
    )
    
    # Rename columns to clear semantic terms
    segments.columns = ['start_time', 'end_time']
    
    return segments

def export_stair_clips(f_sensor: str, f_label: str, output_dir: str) -> pd.DataFrame:
    """
    Slices raw high-frequency IMU sensor data into isolated clips based on 
    continuous 'stairs_up' time boundaries and saves them as separate CSV files.

    Parameters:
    -----------
    f_sensor : str
        The full system path to the raw high-frequency IMU sensor data file 
        (e.g., '/path/to/sensor_data.csv'). This file must contain a 'time' 
        column (in milliseconds) alongside sensor axes data (e.g., acc_x, gyro_z).
        
    f_label : str
        The full system path to the 'labels.csv' ground-truth file. 
        Expected directory structure must match: '.../data_set/[course_name]/[subj_id]/labels.csv'.
        This file must contain at minimum the following two columns:
            - 'time' : timestamp values matching the sensor data scale.
            - 'walk_mode' : strings indicating activity (e.g., 'stairs_up', 'flat').
            
    output_dir : str
        The path to the destination directory where the sliced segment CSVs 
        will be stored (typically 'data/processed/'). The directory will be 
        created automatically if it does not already exist.

    Returns:
    --------
    pd.DataFrame
        The segements pd.DataFrame containing the start and end times of each continuous 
        stairs up segment eg:
           start_time  end_time
        0       24566     37233
        1      115550    130400
        2      249866    263516
        3      342316    356583
        4      481766    495366
        5      573316    587066

        Outputs individual sliced CSV files directly to the disk inside `output_dir` 
        using the naming convention: '[subj_id]_[course_name]_stairsup_clip_[index].csv'.

    Raises:
    -------
    FileNotFoundError
        If either the `f_sensor` or `f_label` paths do not point to valid files.
    """
    # Verify file paths exist before processing heavy datasets
    if not os.path.exists(f_sensor):
        raise FileNotFoundError(f"Raw sensor file not found at: {f_sensor}")
    if not os.path.exists(f_label):
        raise FileNotFoundError(f"Ground-truth label file not found at: {f_label}")
    
    # 1. Get our start/end times for stairs up
    segments = get_stair_segments(f_label, target_mode="stairs_up")
    
    if segments.empty:
        print(f"No 'stairs_up' segments found for {Path(f_label).name}. Skipping.")
        return

    # 2. Extract metadata components for descriptive naming
    path_parts = Path(f_label).parts
    subj_id = path_parts[-2]
    course_name = path_parts[-3]
    
    # 3. Load the heavy, high-frequency raw sensor data
    print(f"Loading raw sensor data: {Path(f_sensor).name}...")
    df_sensor = pd.read_csv(f_sensor)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Loop over segments and save individual continuous clips
    for idx, row in segments.iterrows():
        # Mask the sensor data to strictly keep data within the stair timestamps
        clip = df_sensor[(df_sensor.time >= row['start_time']) & 
                         (df_sensor.time <= row['end_time'])]
        
        # Build a descriptive unique name for this specific staircase instance
        filename = f"{subj_id}_{course_name}_stairsup_clip_{idx}.csv"
        out_path = os.path.join(output_dir, filename)
        
        # Export to a lightweight CSV (dropping index to save space)
        clip.to_csv(out_path, index=False)
        print(f"   Successfully exported: {filename} ({len(clip)} rows)")

    return segments

# Simple test block to make sure it runs when you execute this file directly
if __name__ == "__main__":
    # Example usage mock path
    print("Testing get_stair_segments logic execution...")
    path = r'C:\Users\theil\Documents\0_summer2026\0_git\StairsUp\data\data_set\courseA\id01\labels.csv'
    segments_df = get_stair_segments(path)
    for start, end in zip(segments_df['start_time'], segments_df['end_time']):
        print(start, end)

