import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from src.preprocessing.extractStairs import extract_stairs

DATA_PATH = Path(__file__).resolve().parent.parent / "data"

def run_algorithm


if __name__ == "__main__":
    file_list = list(DATA_PATH.rglob("xsens.csv"))
    print(f"Found {len(file_list)} files to process.")

    if not file_list:
        print("No files found. Exiting.")
        exit()
    
    # Multiprocessing
    with ProcessPoolExecutor() as executor:
        results = executor.map(, file_list)

        for r in results:
            print(r)