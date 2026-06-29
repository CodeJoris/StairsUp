#%% Imports (run first)
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from python_code.src.preprocessing.extractStairs import (
    export_stair_clips,
    get_stair_segments,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_PATH / "processed"

#%% Data Export To only stairs up
file_list = list(DATA_PATH.rglob("labels.csv"))

for file in file_list:
    id = file.parts[-2]
    course = file.parts[-3]

    imu_file = DATA_PATH / "data_set" / course / id / "xsens.csv"

    print(f"exporting {id}_{course}")
    export_stair_clips(imu_file, file, OUTPUT_DIR)
    print(f"exported {id}_{course}")

#%%


#%%
if __name__ == "__main__":
    file_list = list(DATA_PATH.rglob("labels.csv"))
    print(f"Found {len(file_list)} files to process.")

    if not file_list:
        print("No files found. Exiting.")
        exit()
    
    # Multiprocessing
    with ProcessPoolExecutor() as executor:
        results = executor.map(get_stair_segments, file_list)

        for r in results:
            print(r)