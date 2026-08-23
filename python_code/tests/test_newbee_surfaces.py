'''
Test module for the NewBee surfaces.
'''
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_LABEL_COLS = ["time", "walk_mode"]

def get_unique_surface(f_label: str) -> pd.DataFrame:
    """
    Parse a labels.csv file and extract unique surfaces of a walking mode.
    """
    if not os.path.exists(f_label):
        raise FileNotFoundError(f"Label file not found at: {f_label}")

    df = pd.read_csv(f_label, usecols=REQUIRED_LABEL_COLS)

    return set(df["walk_mode"].unique())

if __name__ == "__main__":
    # Example usage
    DATA_PATH = Path(__file__).parent.parent.parent / "data" / 'data_set'
    label_files = list(DATA_PATH.glob("**/labels.csv"))
    unique_surfaces = set()


    for label_file in label_files:
        surfaces = get_unique_surface(str(label_file))
        # print(f'[DEBUG] Surfaces found in {label_file}: {surfaces}')
        len_1 = len(unique_surfaces)
        unique_surfaces.update(surfaces)
        len2 = len(unique_surfaces)
        if len2 > len_1:
            print(f'[DEBUG] Found new surfaces in {label_file}: {surfaces - unique_surfaces}')
        else:
            print(f'[DEBUG] No new surfaces found in {label_file}.')

    print(f'[DEBUG] Final unique surfaces: {unique_surfaces}')