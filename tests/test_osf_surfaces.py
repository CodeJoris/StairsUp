'''
This checks the available surfaces in the osf dataset using 
the gaitmap-datasets library and filters for stair climbing sequences.
'''
from os import name

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from gaitmap_datasets import StairAmbulationHealthy2021PerTest

def extract_available_surfaces(dataset_base_path: str) -> List[Dict[str, Any]]:
    # Ensure include_pressure_data=True to fetch the ground truth
    dataset = StairAmbulationHealthy2021PerTest(
        data_folder=Path(dataset_base_path), 
        include_pressure_data=True
    )
    
    names = set()  # Use a set to avoid duplicates
    
    for datapoint in dataset:
        test_name = str(datapoint.group.test).lower()
        names.add(test_name)

    print(f"\n---> [DEBUG] Found {len(names)} unique test names in the dataset:")
    print(names)

    return names

if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent.parent / "data" / "stair_ambulation_dataset"
    print(f"\n---> [DEBUG] Checking dataset path: {dataset_path}")

    if dataset_path.exists():
        names = extract_available_surfaces(str(dataset_path))

    TARGET_MODE = "stair_up_flat"  # Change this to your desired surface type

    parts = TARGET_MODE.split('_')
    for name in names:
        # print(f"\n---> [DEBUG] Checking if '{TARGET_MODE}' is in test name: '{name}'")
        # print(f"Contains all parts: {all(part in name for part in parts)}")
        if all(part in name for part in parts):
            print(f"\n---> [DEBUG] Found tests for the surface '{TARGET_MODE}' in the dataset: '{name}'")

    