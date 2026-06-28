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


# Simple test block to make sure it runs when you execute this file directly
if __name__ == "__main__":
    # Example usage mock path
    print("Testing get_stair_segments logic execution...")
    # segments_df = get_stair_segments("path/to/your/labels.csv")
    # print(segments_df)