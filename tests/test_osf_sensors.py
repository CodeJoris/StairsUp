'''
Check the existing sensors in the osf 
'''

import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from gaitmap_datasets import StairAmbulationHealthy2021PerTest

def main(dataset_base_path: str) -> List[Dict[str, Any]]:
    # Ensure include_pressure_data=True to fetch the ground truth
    dataset = StairAmbulationHealthy2021PerTest(
        data_folder=Path(dataset_base_path), 
        include_pressure_data=True,
        include_hip_sensor = True
    )

    for datapoint in dataset:
        events = getattr(datapoint, 'pressure_insole_event_list_', None)

        # print(datapoint.data['hip_sensor'])
        print(events)
        print(datapoint.sampling_rate_hz)
        break
                
if __name__ == "__main__":
    dataset_path = Path(__file__).parent.parent.parent / "data" / "osf"
    extracted_tests = main(dataset_path)