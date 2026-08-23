'''
Check the existing sensors in the osf 
'''

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from gaitmap_datasets import StairAmbulationHealthy2021PerTest

def extract_ascending_stairs_data(dataset_base_path: str) -> List[Dict[str, Any]]:
    # Ensure include_pressure_data=True to fetch the ground truth
    dataset = StairAmbulationHealthy2021PerTest(
        data_folder=Path(dataset_base_path), 
        include_pressure_data=True,
        include_hip_sensor = True
    )

    for datapoint in dataset:
        
        # Use the correct attribute for discrete gait events
        events = getattr(datapoint, 'pressure_insole_event_list_', None)
        if events is None or events.empty if hasattr(events, 'empty') else not events:
            continue 

        print(datapoint.data.keys())
        break
                
if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent.parent / "data" / "osf"
    extracted_tests = extract_ascending_stairs_data(dataset_path)